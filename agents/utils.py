from collections import namedtuple

import gym
import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

Transition = namedtuple('Transition', ('obs', 'action', 'reward', 'next_obs', 'done'))
Transition2 = namedtuple('Transition2', ('obs', 'action', 'reward', 'next_obs', 'done', 'info', 'threshold',
                                         'curr_return'))


def stringify(instance):
    name = instance.__class__.__name__
    params = ''
    for k, v in instance.__dict__.items():
        v = str(v)
        params += f'{k}: {v}\n'
    params = '\n'.join(['  ' + x.rstrip() for x in params.split('\n') if len(x.strip()) > 0])
    return f"{name}(\n{params}\n)"


def create_mlp(layer_sizes):
    layers = []
    for i in range(len(layer_sizes) - 1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i + 1]
        layers.append(nn.Linear(input_size, output_size))
        layers.append(nn.Tanh())
    layers.pop()
    return nn.Sequential(*layers)


def create_conv_net(obs_shape, kernel_sizes, paddings, channel_sizes, fc_sizes):
    assert len(obs_shape) == 3
    num_conv_layers = len(kernel_sizes)
    assert len(paddings) == num_conv_layers
    assert len(channel_sizes) == num_conv_layers
    in_channels, width, height = obs_shape
    layers = []
    for i in range(num_conv_layers):
        out_channels = channel_sizes[i]
        kernel_size = kernel_sizes[i]
        padding = paddings[i]
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                padding=padding))
        layers.append(nn.ReLU())
        in_channels = out_channels
        width += -(kernel_size - 1) + 2 * padding
        height += -(kernel_size - 1) + 2 * padding
    layers.append(nn.Flatten())
    input_size = width * height * in_channels
    for output_size in fc_sizes:
        layers.append(nn.Linear(input_size, output_size))
        layers.append(nn.ReLU())
        input_size = output_size
    layers.pop()
    return nn.Sequential(*layers)


class CostThresholdWrapper(gym.Wrapper):
    def __init__(self, env, cost_threshold):
        super().__init__(env)
        self.cost_threshold = cost_threshold
        self._cost = 0.
        self.observation_space = spaces.Box(shape=(env.observation_space.shape[0]+1,), low=-np.inf, high=np.inf)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        self._cost += info['cost']
        cost_frac = self._cost / self.cost_threshold
        cost_frac = min(cost_frac, 1)
        cost_frac = max(cost_frac, 0)
        assert 0 <= cost_frac <= 1
        next_obs = np.concatenate((next_obs, [cost_frac]))
        return next_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._cost = 0.
        return np.concatenate((obs, [0.]))


class SafetyWrapper(gym.Wrapper):
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        info = info or {}
        info['cost'] = 0.
        return next_obs, reward, done, info


class GaussianMlp(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_sizes, max_action, min_variance, max_variance):
        super().__init__()
        assert len(obs_shape) == 1, f'unsupported obs_shape: {obs_shape}'
        assert len(action_shape) == 1, f'unsupported action_shape: {action_shape}'
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.max_action = max_action
        self.min_variance = min_variance
        self.max_variance = max_variance

        body_layer_sizes = [obs_shape[0]] + hidden_sizes
        last_hidden_size = body_layer_sizes[-1]
        action_size = action_shape[0]
        self.body = create_mlp(body_layer_sizes)
        self.mean_head = nn.Linear(last_hidden_size, action_size)
        self.var_head = nn.Linear(last_hidden_size, action_size)

    def forward(self, x):
        bs = x.shape[0]
        assert x.shape == (bs, *self.obs_shape)
        x = torch.tanh(self.body(x))
        mean = torch.tanh(self.mean_head(x)) * self.max_action
        var = F.softplus(self.var_head(x))
        var = torch.clamp(var, min=self.min_variance, max=self.max_variance)
        return mean, var


class CategoricalProjection(nn.Module):
    def __init__(self, v_min, v_max, num_atoms, discount_factor):
        super().__init__()
        assert v_max > v_min
        assert num_atoms > 1
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.discount_factor = discount_factor
        self.atom_delta = (v_max - v_min) / (num_atoms - 1)
        self.atom_values = torch.FloatTensor([v_min + self.atom_delta * i for i in range(self.num_atoms)])

    def forward(self, reward, probs, not_done):
        bs = probs.shape[0]
        assert reward.shape == (bs, 1)
        assert not_done.shape == (bs, 1)
        assert probs.shape == (bs, self.num_atoms)

        atom_values = self.atom_values.unsqueeze(0).expand((bs, self.num_atoms))
        reward = reward.expand((bs, self.num_atoms))
        new_atom_values = reward + self.discount_factor * not_done * atom_values
        new_atom_values = torch.clamp(new_atom_values, min=self.v_min, max=self.v_max)

        # Fast projection borrowed from here:
        # https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent/CategoricalDQN_agent.py#L108
        b = (new_atom_values - self.v_min) / self.atom_delta
        l = b.floor()
        u = b.ceil()
        d_m_l = (u + (l == u).float() - b) * probs
        d_m_u = (b - l) * probs

        new_probs = torch.zeros_like(probs)
        for i in range(bs):
            new_probs[i].index_add_(0, l[i].long(), d_m_l[i])
            new_probs[i].index_add_(0, u[i].long(), d_m_u[i])

        return new_probs


class DiscretePolicy:
    def __init__(self, net, obs_shape, num_actions):
        self.net = net
        self.obs_shape = obs_shape
        self.num_actions = num_actions

    def __str__(self):
        return stringify(self)

    def sample_action(self, obs):
        assert obs.shape == self.obs_shape
        obs = torch.FloatTensor(obs).unsqueeze(0)
        logits = self.net(obs).squeeze(0)
        assert logits.shape == (self.num_actions,)
        distribution = Categorical(logits=logits)
        return distribution.sample().item()

    def distribution_params(self, obs):
        assert obs.shape[1:] == self.obs_shape
        probs = F.softmax(self.net(obs), dim=1)
        return (probs,)

    def _validate_distribution_params(self, distribution_params):
        assert len(distribution_params) == 1
        assert distribution_params[0].shape[1:] == (self.num_actions,)

    def probs(self, distribution_params, action):
        bs = action.shape[0]
        assert action.shape == (bs, 1)
        self._validate_distribution_params(distribution_params)
        action_index = action.squeeze(1)
        assert action_index.shape == (bs,)
        probs = distribution_params[0]
        probs = probs[range(bs), action_index]
        assert probs.shape == (bs,)
        return probs.unsqueeze(1)

    def entropy(self, distribution_params):
        self._validate_distribution_params(distribution_params)
        distribution = Categorical(probs=distribution_params[0])
        return distribution.entropy().mean()

    def kl_divergence(self, distribution_params1, distribution_params2):
        self._validate_distribution_params(distribution_params1)
        self._validate_distribution_params(distribution_params2)
        return torch.distributions.kl.kl_divergence(
            Categorical(probs=distribution_params1[0]),
            Categorical(probs=distribution_params2[0])).mean()


class GaussianPolicy:
    def __init__(self, net, obs_shape, action_shape, action_low, action_high):
        self.net = net
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_low = action_low
        self.action_high = action_high

    def __str__(self):
        return stringify(self)

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def sample_action(self, obs):
        assert obs.shape == self.obs_shape
        obs = torch.FloatTensor(obs).unsqueeze(0)
        assert obs.shape == (1, *self.obs_shape)
        distribution_params = self.distribution_params(obs)
        distribution = self._create_distribution(distribution_params)
        action = distribution.sample()
        assert action.shape == (1, *self.action_shape)
        action = action.squeeze(0).numpy()
        action = np.clip(action, a_max=self.action_high, a_min=self.action_low)
        return action

    def distribution_params(self, obs):
        bs = obs.shape[0]
        assert obs.shape == (bs, *self.obs_shape)
        mean, variance = self.net(obs)
        assert mean.shape == (bs, *self.action_shape)
        assert variance.shape == (bs, *self.action_shape)
        if torch.isnan(variance).sum() > 0:
            print('nan var:', variance)
            print('obs:', obs)
            print('params:')
            for param in self.net.parameters():
                print(param)
        return mean, variance

    def _validate_distribution_params(self, distribution_params):
        assert len(distribution_params) == 2
        mean, variance = distribution_params
        bs = mean.shape[0]
        assert mean.shape == (bs, *self.action_shape)
        assert variance.shape == (bs, *self.action_shape)

    def _create_distribution(self, distribution_params):
        self._validate_distribution_params(distribution_params)
        mean, variance = distribution_params
        bs = mean.shape[0]
        assert mean.shape == (bs, *self.action_shape)
        assert variance.shape == (bs, *self.action_shape)
        covariance = torch.diag_embed(variance)
        dets = torch.det(covariance)
        if not torch.all(dets > 0):
            print('singular covariance matrix detected')
            print('dets:', dets)
            print('covariance:', covariance)
            print('variance:', variance)
        distribution = MultivariateNormal(loc=mean, covariance_matrix=covariance)
        return distribution

    def probs(self, distribution_params, action):
        bs = action.shape[0]
        distribution = self._create_distribution(distribution_params)
        log_probs = distribution.log_prob(action).unsqueeze(1)
        assert log_probs.shape == (bs, 1)
        probs = torch.exp(log_probs)
        assert probs.shape == (bs, 1)
        return probs

    def entropy(self, distribution_params):
        distribution = self._create_distribution(distribution_params)
        return distribution.entropy().mean()

    def kl_divergence(self, distribution_params1, distribution_params2):
        distribution1 = self._create_distribution(distribution_params1)
        distribution2 = self._create_distribution(distribution_params2)
        return torch.distributions.kl.kl_divergence(distribution1, distribution2).mean()


class TD0:
    def __init__(self, net, optimizer, discount_factor, obs_shape):
        self.net = net
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.obs_shape = obs_shape

    def __str__(self):
        return stringify(self)

    def exp_value(self, obs):
        return self.net(obs)

    def advantage(self, obs, next_obs, reward, not_done):
        bs = obs.shape[0]
        assert obs.shape == (bs, *self.obs_shape)
        assert next_obs.shape == (bs, *self.obs_shape)
        assert reward.shape == (bs, 1)
        assert not_done.shape == (bs, 1)

        curr_value = self.net(obs)
        next_value = self.net(next_obs)
        assert curr_value.shape == (bs, 1), f"Expected {(bs, 1)}, found curr_value.shape={curr_value.shape}"
        assert next_value.shape == (bs, 1)
        return reward + self.discount_factor * not_done * next_value - curr_value

    def update(self, obs, reward, next_obs, not_done):
        bs = obs.shape[0]
        assert obs.shape == (bs, *self.obs_shape)
        assert reward.shape == (bs, 1)
        assert next_obs.shape == (bs, *self.obs_shape)
        assert not_done.shape == (bs, 1)

        target_value = (reward + self.discount_factor * not_done * self.net(next_obs)).detach()
        assert target_value.shape == (bs, 1), f'invalid target_value shape: {target_value.shape}'
        pred_value = self.net(obs)
        assert pred_value.shape == (bs, 1)
        loss = F.mse_loss(input=pred_value, target=target_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class CategoricalTD0:
    def __init__(self, net, optimizer, discount_factor, obs_shape, num_atoms, v_min, v_max):
        assert num_atoms > 1
        assert v_max > v_min
        self.net = net
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.obs_shape = obs_shape
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.atom_delta = (v_max - v_min) / (num_atoms - 1)
        self.atom_values = torch.FloatTensor([v_min + self.atom_delta * i for i in range(self.num_atoms)])
        self.projection = CategoricalProjection(v_min=v_min, v_max=v_max, num_atoms=num_atoms,
                                                discount_factor=discount_factor)

    def __str__(self):
        return stringify(self)

    def exp_value(self, obs):
        bs = obs.shape[0]
        assert obs.shape == (bs, *self.obs_shape)
        logits = self.net(obs)
        assert logits.shape == (bs, self.num_atoms)
        probs = F.softmax(logits, dim=1)
        exp_value = (probs * self.atom_values).sum(dim=1).unsqueeze(1)
        assert exp_value.shape == (bs, 1)
        return exp_value

    def probs(self, obs):
        bs = obs.shape[0]
        assert obs.shape == (bs, *self.obs_shape)
        logits = self.net(obs)
        assert logits.shape == (bs, self.num_atoms)
        probs = F.softmax(logits, dim=1)
        return probs

    def advantage(self, obs, next_obs, reward, not_done):
        bs = obs.shape[0]
        assert obs.shape == (bs, *self.obs_shape)
        assert next_obs.shape == (bs, *self.obs_shape)
        assert reward.shape == (bs, 1)
        assert not_done.shape == (bs, 1)

        curr_exp_value = self.exp_value(obs)
        next_exp_value = self.exp_value(next_obs)
        assert curr_exp_value.shape == (bs, 1)
        assert next_exp_value.shape == (bs, 1)
        return reward + self.discount_factor * not_done * next_exp_value - curr_exp_value

    def update(self, obs, reward, next_obs, not_done):
        bs = obs.shape[0]
        assert obs.shape == (bs, *self.obs_shape)
        assert reward.shape == (bs, 1)
        assert next_obs.shape == (bs, *self.obs_shape)
        assert not_done.shape == (bs, 1)

        next_value_logits = self.net(next_obs)
        assert next_value_logits.shape == (bs, self.num_atoms)
        next_value_probs = F.softmax(next_value_logits, dim=1)
        target_probs = self.projection(reward=reward, probs=next_value_probs, not_done=not_done).detach()
        assert target_probs.shape == (bs, self.num_atoms)
        curr_value_logits = self.net(obs)
        assert curr_value_logits.shape == (bs, self.num_atoms)
        loss = -(F.log_softmax(curr_value_logits, dim=1) * target_probs).sum(dim=1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
