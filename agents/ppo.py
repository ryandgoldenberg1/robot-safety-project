import argparse
import copy
import json

import gym
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import ai_safety_gridworlds
import agents.utils as utils


class DiscretePolicy:
    def __init__(self, net, obs_shape, num_actions):
        self.net = net
        self.obs_shape = obs_shape
        self.num_actions = num_actions

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


class TD0:
    def __init__(self, net, optimizer, discount_factor, obs_shape):
        self.net = net
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.obs_shape = obs_shape

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
        assert target_value.shape == (bs, 1)
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

    def _exp_value(self, obs):
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

        curr_exp_value = self._exp_value(obs)
        next_exp_value = self._exp_value(next_obs)
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
                 num_train_iters, steps_per_iter, num_actor_epochs, num_critic_epochs, batch_size):
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

        self._obs = self.env.reset()
        self._obs_shape = self._obs.shape
        self._curr_return = 0.

    def train(self):
        for train_iter in range(1, self.num_train_iters + 1):
            transitions, exp_metrics = self.gather_experience()
            update_metrics = self.update(transitions)
            metrics = {**exp_metrics, **update_metrics}
            self.log_metrics(train_iter, metrics)

    def gather_experience(self):
        transitions = []
        returns = []
        for _ in range(self.steps_per_iter):
            action = self.actor.sample_action(self._obs)
            next_obs, reward, done, _ = self.env.step(action)
            transition = utils.Transition(obs=self._obs, action=action, reward=reward, next_obs=next_obs, done=done)
            transitions.append(transition)
            self._obs = self.env.reset() if done else next_obs
            self._curr_return += reward
            if done:
                returns.append(self._curr_return)
                self._curr_return = 0
        avg_return = sum(returns) / len(returns)
        metrics = {'return': avg_return}
        return transitions, metrics

    def _create_dataset(self, transitions):
        obs = torch.FloatTensor([x.obs for x in transitions])
        action = torch.LongTensor([x.action for x in transitions]).unsqueeze(1)
        reward = torch.FloatTensor([x.reward for x in transitions]).unsqueeze(1)
        next_obs = torch.FloatTensor([x.next_obs for x in transitions])
        not_done = torch.FloatTensor([0. if x.done else 1. for x in transitions]).unsqueeze(1)
        num_transitions = len(transitions)
        assert obs.shape == (num_transitions, *self._obs_shape)
        assert next_obs.shape == (num_transitions, *self._obs_shape)
        assert action.shape == (num_transitions, 1)
        assert reward.shape == (num_transitions, 1)
        assert not_done.shape == (num_transitions, 1)

        advantage = self.critic.advantage(obs=obs, next_obs=next_obs, reward=reward, not_done=not_done).detach()
        assert advantage.shape == (num_transitions, 1)
        distribution_params = (x.detach() for x in self.actor.distribution_params(obs))
        return torch.utils.data.TensorDataset(obs, action, reward, next_obs, not_done, advantage, *distribution_params)

    def update(self, transitions):
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

    def log_metrics(self, train_iter, metrics):
        metrics_strs = [f'{k}: {v:.04f}' for k, v in metrics.items()]
        metrics_str = ' | '.join(metrics_strs)
        step = train_iter * self.steps_per_iter
        print(f"[Iter {train_iter} Step {step/1000}k] {metrics_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='CartPole-v1')
    parser.add_argument('--actor_hidden_sizes', nargs='*', type=int, default=[32])
    parser.add_argument('--critic_hidden_sizes', nargs='*', type=int, default=[32])
    parser.add_argument('--actor_learning_rate', type=float, default=0.0001)
    parser.add_argument('--critic_learning_rate', type=float, default=0.0005)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.5)
    parser.add_argument('--kl_coef', type=float, default=0.)
    parser.add_argument('--target_kl', type=float, default=0.05)
    parser.add_argument('--num_train_iters', type=int, default=50)
    parser.add_argument('--steps_per_iter', type=int, default=4000)
    parser.add_argument('--num_actor_epochs', type=int, default=20)
    parser.add_argument('--num_critic_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=128)
    # Categorical Arguments
    parser.add_argument('--use_categorical', default=False, action='store_true')
    parser.add_argument('--num_atoms', type=int, default=51)
    parser.add_argument('--v_min', type=float, default=0.)
    parser.add_argument('--v_max', type=float, default=100.)
    # Convolutional Layers
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[5, 5, 3])
    parser.add_argument('--paddings', type=int, nargs='+', default=[2, 0, 0])
    parser.add_argument('--channel_sizes', type=int, nargs='+', default=[16, 32, 32])
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    env = gym.make(args.env_name)
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    print('obs_shape:', obs_shape)

    policy_layer_sizes = args.actor_hidden_sizes + [num_actions]
    critic_layer_sizes = args.critic_hidden_sizes + [1]
    if args.use_categorical:
        critic_layer_sizes[-1] = args.num_atoms
    if len(obs_shape) == 3:
        conv_kwargs = {'obs_shape': obs_shape, 'kernel_sizes': args.kernel_sizes, 'channel_sizes': args.channel_sizes,
                       'paddings': args.paddings}
        policy_net = utils.create_conv_net(fc_sizes=policy_layer_sizes, **conv_kwargs)
        critic_net = utils.create_conv_net(fc_sizes=critic_layer_sizes, **conv_kwargs)
    else:
        policy_layer_sizes = [obs_shape[0]] + policy_layer_sizes
        critic_layer_sizes = [obs_shape[0]] + critic_layer_sizes
        policy_net = utils.create_mlp(policy_layer_sizes)
        critic_net = utils.create_mlp(critic_layer_sizes)
    print('policy_net:', policy_net)
    print('critic_net:', critic_net)

    actor = DiscretePolicy(net=policy_net, obs_shape=obs_shape, num_actions=num_actions)
    actor_optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=args.critic_learning_rate)
    if args.use_categorical:
        critic = CategoricalTD0(net=critic_net, optimizer=critic_optimizer, discount_factor=args.discount_factor,
                                obs_shape=obs_shape, num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max)
    else:
        critic = TD0(net=critic_net, optimizer=critic_optimizer, discount_factor=args.discount_factor,
                     obs_shape=obs_shape)

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
        batch_size=args.batch_size
    )
    ppo.train()


if __name__ == '__main__':
    main()
