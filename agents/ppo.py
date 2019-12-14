import argparse
import copy
import json
import os

import gym
import gym.spaces as spaces
import numpy as np
import torch

import agents.utils as utils
import ai_safety_gridworlds  # pylint: disable=unused-import
import safety_gym  # pylint: disable=unused-import


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
        return utils.DiscretePolicy(net=policy_net, obs_shape=obs_shape, num_actions=num_actions)

    assert len(obs_shape) == 1, f'unsupported obs_shape: {obs_shape}'
    if isinstance(env.action_space, spaces.Discrete):
        num_actions = env.action_space.n
        policy_layer_sizes = [obs_shape[0]] + args.actor_hidden_sizes + [num_actions]
        policy_net = utils.create_mlp(policy_layer_sizes)
        return utils.DiscretePolicy(net=policy_net, obs_shape=obs_shape, num_actions=num_actions)

    action_shape = env.action_space.shape
    assert len(action_shape) == 1
    max_action_high = np.abs(env.action_space.high).max()
    max_action_low = np.abs(env.action_space.low).max()
    assert not np.isinf(max_action_high) and not np.isinf(max_action_low)
    max_action = max(max_action_high.item(), max_action_low.item())
    policy_net = utils.GaussianMlp(obs_shape=obs_shape, action_shape=action_shape,
                                   hidden_sizes=args.actor_hidden_sizes, max_action=max_action,
                                   min_variance=args.min_variance, max_variance=args.max_variance)
    return utils.GaussianPolicy(net=policy_net, obs_shape=obs_shape, action_shape=action_shape,
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
        return utils.CategoricalTD0(net=critic_net, optimizer=critic_optimizer, discount_factor=args.discount_factor,
                                    obs_shape=obs_shape, num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max)

    return utils.TD0(net=critic_net, optimizer=critic_optimizer, discount_factor=args.discount_factor,
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
    parser.add_argument('--seed', type=int)
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

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        env.seed(args.seed)

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
