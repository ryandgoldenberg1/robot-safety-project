import argparse
import copy
import json
import os

import gym
import gym.spaces as spaces
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

import agents.utils as utils
import ai_safety_gridworlds  # pylint: disable=unused-import


class DiscretePolicy:
    def __init__(self, net, obs_shape, num_actions):
        self.net = net
        self.obs_shape = obs_shape
        self.num_actions = num_actions

    def __str__(self):
        return utils.stringify(self)

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
        return utils.stringify(self)

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
        return utils.stringify(self)

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
        self.projection = utils.CategoricalProjection(v_min=v_min, v_max=v_max, num_atoms=num_atoms,
                                                      discount_factor=discount_factor)

    def __str__(self):
        return utils.stringify(self)

    def exp_value(self, obs):
        bs = obs.shape[0]
        assert obs.shape == (bs, *self.obs_shape)
        logits = self.net(obs)
        assert logits.shape == (bs, self.num_atoms)
        probs = F.softmax(logits, dim=1)
        exp_value = (probs * self.atom_values).sum(dim=1).unsqueeze(1)
        assert exp_value.shape == (bs, 1)
        return exp_value

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


class PPO:
    def __init__(self, env, actor, critic, actor_optimizer, clip_epsilon, entropy_coef, kl_coef, target_kl,
                 num_train_iters, steps_per_iter, num_actor_epochs, num_critic_epochs, batch_size,
                 save_interval, run_dir):
        if save_interval is not None:
            assert run_dir is not None
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.target_kl = target_kl
        self.num_train_iters = num_train_iters
        self.steps_per_iter = steps_per_iter
        self.num_actor_epochs = num_actor_epochs
        self.num_critic_epochs = num_critic_epochs
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.run_dir = run_dir

        self._obs = self.env.reset()
        self._obs_shape = self._obs.shape
        self._is_action_discrete = isinstance(env.action_space, spaces.Discrete)
        self._action_shape = (1,) if self._is_action_discrete else env.action_space.shape
        self._curr_return = 0.
        self._metrics = []
        if self.run_dir is not None:
            self.models_dir = os.path.join(self.run_dir, 'models')
            os.makedirs(self.models_dir, exist_ok=True)

    def train(self):
        for train_iter in range(1, self.num_train_iters + 1):
            transitions, exp_metrics = self.gather_experience()
            update_metrics = self.update(transitions)
            metrics = {**exp_metrics, **update_metrics}
            self.log_metrics(train_iter, metrics)

    def gather_experience(self):
        transitions = []
        returns = []
        unsafe_actions = 0
        for _ in range(self.steps_per_iter):
            action = self.actor.sample_action(self._obs)
            next_obs, reward, done, info = self.env.step(action)
            transition = utils.Transition(obs=self._obs, action=action, reward=reward, next_obs=next_obs, done=done)
            transitions.append(transition)
            self._obs = self.env.reset() if done else next_obs
            self._curr_return += reward
            if done:
                returns.append(self._curr_return)
                self._curr_return = 0
            if info is not None and info.get('unsafe') is True:
                unsafe_actions += 1
        avg_return = sum(returns) / len(returns)
        metrics = {'return': avg_return, 'unsafe_actions': unsafe_actions}
        return transitions, metrics

    def _create_dataset(self, transitions):
        obs = torch.FloatTensor([x.obs for x in transitions])
        if self._is_action_discrete:
            action = torch.LongTensor([x.action for x in transitions]).unsqueeze(1)
        else:
            action = torch.FloatTensor([x.action for x in transitions])
        reward = torch.FloatTensor([x.reward for x in transitions]).unsqueeze(1)
        next_obs = torch.FloatTensor([x.next_obs for x in transitions])
        not_done = torch.FloatTensor([0. if x.done else 1. for x in transitions]).unsqueeze(1)
        num_transitions = len(transitions)
        assert obs.shape == (num_transitions, *self._obs_shape)
        assert next_obs.shape == (num_transitions, *self._obs_shape)
        assert action.shape == (num_transitions, *self._action_shape), f'invalid action shape: {action.shape}'
        assert reward.shape == (num_transitions, 1)
        assert not_done.shape == (num_transitions, 1)

        advantage = self.critic.advantage(obs=obs, next_obs=next_obs, reward=reward, not_done=not_done).detach()
        assert advantage.shape == (num_transitions, 1)
        distribution_params = (x.detach() for x in self.actor.distribution_params(obs))
        return torch.utils.data.TensorDataset(obs, action, reward, next_obs, not_done, advantage, *distribution_params)

    def update(self, transitions):
        # Create Dataset
        dataset = self._create_dataset(transitions)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        metrics = {}

        # Update Critic
        for _ in range(1, self.num_critic_epochs + 1):
            critic_losses = []
            for (obs, action, reward, next_obs, not_done, *_) in data_loader:
                critic_loss = self.critic.update(obs=obs, reward=reward, next_obs=next_obs, not_done=not_done)
                critic_losses.append(critic_loss * obs.shape[0])
            metrics['critic_loss'] = sum(critic_losses) / len(dataset)

        # Update Actor
        best_actor_state = copy.deepcopy(self.actor.net.state_dict())
        metrics['best_epoch'] = 0
        for epoch in range(1, self.num_actor_epochs + 1):
            actor_losses = []
            entropies = []
            kl_divergences = []
            for (obs, action, reward, next_obs, not_done, advantage, *old_distribution_params) in data_loader:
                bs = obs.shape[0]
                assert advantage.shape == (bs, 1)
                # Normalize Advantage
                advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-5)).detach()
                old_probs = self.actor.probs(old_distribution_params, action)
                assert old_probs.shape == (bs, 1)
                new_distribution_params = self.actor.distribution_params(obs)
                new_probs = self.actor.probs(new_distribution_params, action)
                assert new_probs.shape == (bs, 1)
                prob_ratio = new_probs / old_probs
                prob_ratio_clip = torch.clamp(prob_ratio, min=1 - self.clip_epsilon, max=1 + self.clip_epsilon)
                clip_loss = -torch.min(prob_ratio * advantage, prob_ratio_clip * advantage).mean()
                entropy = self.actor.entropy(new_distribution_params)
                kl_divergence = self.actor.kl_divergence(old_distribution_params, new_distribution_params)
                actor_loss = clip_loss - self.entropy_coef * entropy + self.kl_coef * kl_divergence
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                actor_losses.append(actor_loss.item() * bs)
                entropies.append(entropy.item() * bs)
                kl_divergences.append(kl_divergence.item() * bs)
            avg_actor_loss = sum(actor_losses) / len(dataset)
            avg_entropy = sum(entropies) / len(dataset)
            avg_kl = sum(kl_divergences) / len(dataset)
            if avg_kl <= self.target_kl:
                best_actor_state = copy.deepcopy(self.actor.net.state_dict())
                metrics = {**metrics, 'actor_loss': avg_actor_loss, 'entropy': avg_entropy, 'kl': avg_kl,
                           'best_epoch': epoch}
        self.actor.net.load_state_dict(best_actor_state)

        return metrics

    def _save_models(self, path):
        torch.save({
            'actor': self.actor.net.state_dict(),
            'critic': self.critic.net.state_dict(),
        }, path)

    def log_metrics(self, train_iter, metrics):
        metrics_strs = [f'{k}: {v:.04f}' for k, v in metrics.items()]
        metrics_str = ' | '.join(metrics_strs)
        step = train_iter * self.steps_per_iter
        print(f"[Iter {train_iter} Step {step/1000}k] {metrics_str}")
        metrics = {'train_iter': train_iter, 'step': step, **metrics}
        self._metrics.append(metrics)
        # Save model at regular interval
        if self.save_interval is not None and train_iter % self.save_interval == 0:
            save_path = os.path.join(self.models_dir, f'step-{step}.pt')
            self._save_models(save_path)
            print(f'  Wrote checkpoint to: {save_path}')
        # Save final model and run metrics at the end
        if train_iter == self.num_train_iters and self.run_dir is not None:
            model_path = os.path.join(self.run_dir, f'final-model-{step}.pt')
            self._save_models(model_path)
            print(f'  Wrote final model to: {model_path}')
            metrics_path = os.path.join(self.run_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self._metrics, f)
            print(f'  Wrote run metrics to: {metrics_path}')


def _create_policy(env, args):
    obs_shape = env.observation_space.shape

    if len(obs_shape) == 3:
        assert isinstance(env.action_space, spaces.Discrete), f'unsupport action_space: {env.action_space}'
        conv_kwargs = {'obs_shape': obs_shape, 'kernel_sizes': args.kernel_sizes, 'channel_sizes': args.channel_sizes,
                       'paddings': args.paddings}
        num_actions = env.action_space.n
        policy_layer_sizes = args.actor_hidden_sizes + [num_actions]
        policy_net = utils.create_conv_net(fc_sizes=policy_layer_sizes, **conv_kwargs)
        return DiscretePolicy(net=policy_net, obs_shape=obs_shape, num_actions=num_actions)

    assert len(obs_shape) == 1, f'unsupported obs_shape: {obs_shape}'
    if isinstance(env.action_space, spaces.Discrete):
        num_actions = env.action_space.n
        policy_layer_sizes = [obs_shape[0]] + args.actor_hidden_sizes + [num_actions]
        policy_net = utils.create_mlp(policy_layer_sizes)
        return DiscretePolicy(net=policy_net, obs_shape=obs_shape, num_actions=num_actions)

    action_shape = env.action_space.shape
    assert len(action_shape) == 1
    max_action_high = np.abs(env.action_space.high).max()
    max_action_low = np.abs(env.action_space.low).max()
    assert not np.isinf(max_action_high) and not np.isinf(max_action_low)
    max_action = max(max_action_high.item(), max_action_low.item())
    policy_net = utils.GaussianMlp(obs_shape=obs_shape, action_shape=action_shape,
                                   hidden_sizes=args.actor_hidden_sizes, max_action=max_action,
                                   min_variance=args.min_variance, max_variance=args.max_variance)
    return GaussianPolicy(net=policy_net, obs_shape=obs_shape, action_shape=action_shape,
                          action_low=env.action_space.low, action_high=env.action_space.high)


def _create_critic(env, args):
    obs_shape = env.observation_space.shape
    last_layer_size = 1
    if args.use_categorical:
        last_layer_size = args.num_atoms

    if len(obs_shape) == 3:
        assert isinstance(env.action_space, spaces.Discrete), f'unsupported action_space: {env.action_space}'
        conv_kwargs = {'obs_shape': obs_shape, 'kernel_sizes': args.kernel_sizes, 'channel_sizes': args.channel_sizes,
                       'paddings': args.paddings}
        critic_layer_sizes = args.critic_hidden_sizes + [last_layer_size]
        critic_net = utils.create_conv_net(fc_sizes=critic_layer_sizes, **conv_kwargs)
    else:
        assert len(obs_shape) == 1, f'unsupport obs_shape: {obs_shape}'
        critic_layer_sizes = [obs_shape[0]] + args.critic_hidden_sizes + [last_layer_size]
        critic_net = utils.create_mlp(critic_layer_sizes)

    critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=args.critic_learning_rate)
    if args.use_categorical:
        return CategoricalTD0(net=critic_net, optimizer=critic_optimizer, discount_factor=args.discount_factor,
                              obs_shape=obs_shape, num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max)

    return TD0(net=critic_net, optimizer=critic_optimizer, discount_factor=args.discount_factor,
               obs_shape=obs_shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='CartPole-v1')
    parser.add_argument('--actor_hidden_sizes', nargs='*', type=int, default=[64, 64])
    parser.add_argument('--critic_hidden_sizes', nargs='*', type=int, default=[64, 64])
    parser.add_argument('--actor_learning_rate', type=float, default=0.0003)
    parser.add_argument('--actor_weight_decay', type=float, default=0.)
    parser.add_argument('--critic_learning_rate', type=float, default=0.01)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.)
    parser.add_argument('--kl_coef', type=float, default=0.)
    parser.add_argument('--target_kl', type=float, default=0.03)
    parser.add_argument('--num_train_iters', type=int, default=50)
    parser.add_argument('--steps_per_iter', type=int, default=4000)
    parser.add_argument('--num_actor_epochs', type=int, default=80)
    parser.add_argument('--num_critic_epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--save_interval', type=int)
    parser.add_argument('--run_dir')
    # Categorical Arguments
    parser.add_argument('--use_categorical', default=False, action='store_true')
    parser.add_argument('--num_atoms', type=int, default=51)
    parser.add_argument('--v_min', type=float, default=0.)
    parser.add_argument('--v_max', type=float, default=100.)
    # Convolutional Layers
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[5, 5, 3])
    parser.add_argument('--paddings', type=int, nargs='+', default=[2, 0, 0])
    parser.add_argument('--channel_sizes', type=int, nargs='+', default=[16, 32, 32])
    # Continuous Action Space
    parser.add_argument('--min_variance', type=float, default=1e-6)
    parser.add_argument('--max_variance', type=float, default=float('inf'))
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    if args.run_dir is not None:
        os.makedirs(args.run_dir, exist_ok=True)
        args_path = os.path.join(args.run_dir, 'args.json')
        with open(args_path, 'w') as f:
            json.dump(args.__dict__, f)

    env = gym.make(args.env_name)
    actor = _create_policy(env, args)
    critic = _create_critic(env, args)
    print(actor)
    print(critic)
    actor_optimizer = torch.optim.Adam(actor.net.parameters(), lr=args.actor_learning_rate,
                                       weight_decay=args.actor_weight_decay)

    ppo = PPO(
        env=env,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        kl_coef=args.kl_coef,
        target_kl=args.target_kl,
        num_train_iters=args.num_train_iters,
        steps_per_iter=args.steps_per_iter,
        num_actor_epochs=args.num_actor_epochs,
        num_critic_epochs=args.num_critic_epochs,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        run_dir=args.run_dir
    )
    ppo.train()


if __name__ == '__main__':
    main()
