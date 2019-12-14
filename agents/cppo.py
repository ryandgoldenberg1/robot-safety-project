import argparse
import copy
import json
import os
import time

import gym
import gym.wrappers as wrappers
import numpy as np
import torch

import agents.risk as risk
import agents.utils as utils
import safety_gym  # pylint: disable=unused-import
from safety_gym.envs.engine import Engine


class CTD0:
    def __init__(self, reward_net, cost_net, reward_optimizer, cost_optimizer, discount_factor, obs_shape, num_atoms,
                 cost_min, cost_max, risk_measure, risk_schedule):
        self.reward_net = reward_net
        self.cost_net = cost_net
        self.reward_td0 = utils.TD0(net=reward_net, optimizer=reward_optimizer, discount_factor=discount_factor,
                                    obs_shape=obs_shape)
        self.cost_td0 = utils.CategoricalTD0(net=cost_net, optimizer=cost_optimizer, discount_factor=discount_factor,
                                             obs_shape=obs_shape, num_atoms=num_atoms, v_min=cost_min, v_max=cost_max)
        self.risk_measure = risk_measure
        self.risk_schedule = risk_schedule

    def __str__(self):
        return utils.stringify(self)

    def save(self, path):
        torch.save({'reward': self.reward_net.state_dict(), 'cost': self.cost_net.state_dict()}, path)

    def update(self, obs, next_obs, reward, cost, not_done):
        reward_loss = self.reward_td0.update(obs=obs, next_obs=next_obs, reward=reward, not_done=not_done)
        cost_loss = self.cost_td0.update(obs=obs, reward=cost, next_obs=next_obs, not_done=not_done)
        return reward_loss, cost_loss

    def advantage(self, obs, next_obs, reward, cost, not_done, threshold, curr_return, train_iter):  # pylint: disable=W0613
        # Reward Advantage
        bs = obs.shape[0]
        reward_advantage = self.reward_td0.advantage(obs=obs, next_obs=next_obs, reward=reward, not_done=not_done)
        assert reward_advantage.shape == (bs, 1)

        # Measure Risk
        risk_coef = self.risk_schedule.get(train_iter)
        atom_values = self.cost_td0.atom_values.clone()
        curr_atom_probs = self.cost_td0.probs(obs)
        next_atom_probs = self.cost_td0.probs(next_obs)
        risk_advantage = self.risk_measure(atom_values=atom_values, curr_atom_probs=curr_atom_probs,
                                           next_atom_probs=next_atom_probs, cost=cost, not_done=not_done,
                                           threshold=threshold, curr_return=curr_return)
        assert risk_advantage.shape == (bs, 1)

        # Combine
        advantage = reward_advantage - risk_coef * risk_advantage
        return advantage, reward_advantage, risk_advantage


class CPPOHyperparameters:
    def __init__(self, num_train_iters, steps_per_iter, save_interval, run_dir, cost_threshold, batch_size,
                 num_actor_epochs, num_critic_epochs, clip_epsilon, entropy_coef, kl_coef, target_kl):
        if save_interval is not None:
            assert run_dir is not None
        self.num_train_iters = num_train_iters
        self.steps_per_iter = steps_per_iter
        self.save_interval = save_interval
        self.run_dir = run_dir
        self.cost_threshold = cost_threshold
        self.batch_size = batch_size
        self.num_actor_epochs = num_actor_epochs
        self.num_critic_epochs = num_critic_epochs
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.target_kl = target_kl

    def __str__(self):
        return utils.stringify(self)


class CPPO:
    def __init__(self, env, actor, actor_optimizer, critic, hp):
        self.env = env
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.critic = critic
        self.hp = hp

        self._obs = self.env.reset()
        self._curr_return = 0.
        self._curr_cost = 0.
        self._obs_shape = self.env.observation_space.shape
        self._action_shape = self.env.action_space.shape

    def __str__(self):
        return utils.stringify(self)

    def train(self):
        for train_iter in range(1, self.hp.num_train_iters + 1):
            start_time = time.time()
            transitions, exp_metrics = self.gather_experience()
            update_metrics = self.update(transitions, train_iter)
            elapsed_time = time.time() - start_time
            metrics = {**exp_metrics, **update_metrics, 'elapsed': elapsed_time}
            self.log(train_iter, metrics)

    def gather_experience(self):
        transitions = []
        returns = []
        costs = []
        violations = []
        for _ in range(self.hp.steps_per_iter):
            action = self.actor.sample_action(self._obs)
            next_obs, reward, done, info = self.env.step(action)
            cost = info['cost']
            threshold = self.hp.cost_threshold - self._curr_cost
            transition = utils.Transition2(obs=self._obs, action=action, reward=reward, next_obs=next_obs, done=done,
                                           info=info, threshold=threshold, curr_return=self._curr_return)
            transitions.append(transition)
            self._obs = self.env.reset() if done else next_obs
            self._curr_return += reward
            self._curr_cost += cost
            if done:
                returns.append(self._curr_return)
                costs.append(self._curr_cost)
                violations.append(1. if self._curr_cost >= self.hp.cost_threshold else 0.)
                self._curr_return = 0.
                self._curr_cost = 0.
            metrics = {}
            if len(returns) > 0:
                metrics['episodes'] = len(returns)
                metrics['avg_return'] = sum(returns) / len(returns)
                metrics['avg_cost'] = sum(costs) / len(costs)
                metrics['violation'] = sum(violations) / len(violations)
        return transitions, metrics

    def create_dataset(self, transitions, train_iter):
        obs = torch.FloatTensor([x.obs for x in transitions])
        action = torch.FloatTensor([x.action for x in transitions])
        reward = torch.FloatTensor([x.reward for x in transitions]).unsqueeze(1)
        cost = torch.FloatTensor([x.info['cost'] for x in transitions]).unsqueeze(1)
        next_obs = torch.FloatTensor([x.next_obs for x in transitions])
        not_done = torch.FloatTensor([0. if x.done else 1. for x in transitions]).unsqueeze(1)
        threshold = torch.FloatTensor([x.threshold for x in transitions]).unsqueeze(1)
        curr_return = torch.FloatTensor([x.curr_return for x in transitions]).unsqueeze(1)
        num_transitions = len(transitions)
        assert obs.shape == (num_transitions, *self._obs_shape)
        assert action.shape == (num_transitions, *self._action_shape)
        assert reward.shape == (num_transitions, 1)
        assert cost.shape == (num_transitions, 1)
        assert next_obs.shape == (num_transitions, *self._obs_shape)
        assert not_done.shape == (num_transitions, 1)
        assert threshold.shape == (num_transitions, 1)

        advantage, reward_advantage, risk_advantage = self.critic.advantage(
            obs=obs, next_obs=next_obs, reward=reward, cost=cost, not_done=not_done, threshold=threshold,
            curr_return=curr_return, train_iter=train_iter)
        advantage.detach_()
        distribution_params = (x.detach() for x in self.actor.distribution_params(obs))
        dataset = torch.utils.data.TensorDataset(obs, action, reward, cost, next_obs, not_done, advantage,
                                                 *distribution_params)
        metrics = {}
        metrics['risk_coef'] = self.critic.risk_schedule.get(train_iter)
        metrics['advantage'] = advantage.mean().item()
        metrics['advantage_abs'] = advantage.abs().mean().item()
        metrics['reward_advantage'] = reward_advantage.mean().item()
        metrics['reward_advantage_abs'] = reward_advantage.abs().mean().item()
        metrics['risk_advantage'] = risk_advantage.mean().item()
        metrics['risk_advantage_abs'] = risk_advantage.abs().mean().item()
        return dataset, metrics

    def update(self, transitions, train_iter):
        dataset, metrics = self.create_dataset(transitions, train_iter)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.hp.batch_size, shuffle=True)

        # Update Critic
        reward_losses = []
        cost_losses = []
        for _ in range(1, self.hp.num_critic_epochs + 1):
            reward_loss = 0.
            cost_loss = 0.
            for (obs, action, reward, cost, next_obs, not_done, advantage, *distribution_params) in data_loader:  # pylint: disable=W0612,W0613
                reward_batch_loss, cost_batch_loss = self.critic.update(obs=obs, reward=reward, cost=cost,
                                                                        next_obs=next_obs, not_done=not_done)
                reward_loss += reward_batch_loss * obs.shape[0]
                cost_loss += cost_batch_loss * obs.shape[0]
            reward_loss /= len(dataset)
            cost_loss /= len(dataset)
            reward_losses.append(reward_loss)
            cost_losses.append(cost_loss)
        metrics['avg_reward_loss'] = sum(reward_losses) / len(reward_losses)
        metrics['start_reward_loss'] = reward_losses[0]
        metrics['end_reward_loss'] = reward_losses[-1]
        metrics['avg_cost_loss'] = sum(cost_losses) / len(cost_losses)
        metrics['start_cost_loss'] = cost_losses[0]
        metrics['end_cost_loss'] = cost_losses[-1]

        # Update Actor
        best_actor_state = copy.deepcopy(self.actor.net.state_dict())
        metrics['best_epoch'] = 0
        for epoch in range(1, self.hp.num_actor_epochs + 1):
            epoch_loss = 0.
            epoch_kl = 0.
            epoch_entropy = 0.
            for (obs, action, reward, cost, next_obs, not_done, advantage, *old_distribution_params) in data_loader:
                bs = obs.shape[0]
                advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-5)).detach()
                old_probs = self.actor.probs(old_distribution_params, action).detach()
                assert old_probs.shape == (bs, 1)
                new_distribution_params = self.actor.distribution_params(obs)
                new_probs = self.actor.probs(new_distribution_params, action)
                assert new_probs.shape == (bs, 1)
                prob_ratio = new_probs / old_probs
                prob_ratio_clip = torch.clamp(prob_ratio, min=1 - self.hp.clip_epsilon, max=1 + self.hp.clip_epsilon)
                clip_loss = -torch.min(prob_ratio * advantage, prob_ratio_clip * advantage).mean()
                entropy = self.actor.entropy(new_distribution_params)
                kl_divergence = self.actor.kl_divergence(old_distribution_params, new_distribution_params)
                actor_loss = clip_loss - self.hp.entropy_coef * entropy + self.hp.kl_coef * kl_divergence
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                epoch_loss += actor_loss.item() * bs
                epoch_kl += kl_divergence.item() * bs
                epoch_entropy += entropy.item() * bs
            epoch_loss /= len(dataset)
            epoch_kl /= len(dataset)
            epoch_entropy /= len(dataset)
            if epoch_kl <= self.hp.target_kl:
                best_actor_state = copy.deepcopy(self.actor.net.state_dict())
                metrics['best_epoch'] = epoch
                metrics['kl'] = epoch_kl
                metrics['entropy'] = epoch_entropy
                metrics['actor_loss'] = epoch_loss
            if epoch == 1:
                metrics['start_actor_loss'] = epoch_loss
        self.actor.net.load_state_dict(best_actor_state)
        return metrics

    def log(self, train_iter, metrics):
        step = train_iter * self.hp.steps_per_iter
        header_str = f"[Iter {train_iter:03d} Step {step/1000}k]"
        metrics_strs = [f'{k}: {v:0.03f}' for k, v in metrics.items()]

        log_str = ""
        chunk_size = 10
        for i in range(0, len(metrics_strs), chunk_size):
            header = " " * len(header_str) if i > 0 else header_str
            metrics_str = " | ".join(metrics_strs[i:i+chunk_size])
            log_str += f"{header} {metrics_str}\n"
        log_str = log_str[:-1]
        print(log_str)


        # metrics_str = ' | '.join(metrics_strs)
        # print(f"[Iter {train_iter:03d} Step {step/1000}k] {metrics_str}")
        metrics = {'train_iter': train_iter, 'step': step, **metrics}
        if self.hp.run_dir is not None:
            metrics_path = os.path.join(self.hp.run_dir, 'metrics.json')
            with open(metrics_path, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
        write_last_iter = (train_iter == self.hp.num_train_iters and self.hp.run_dir is not None)
        is_save_iter = (self.hp.save_interval is not None and train_iter % self.hp.save_interval == 0)
        if is_save_iter or write_last_iter:
            save_dir = os.path.join(self.hp.run_dir, f'ckpt-{train_iter:03d}')
            self.save(save_dir)

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.actor.save(os.path.join(save_dir, 'actor.pt'))
        self.critic.save(os.path.join(save_dir, 'critic.pt'))


def create_actor(env, hidden_sizes, min_variance, max_variance):
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_low = env.action_space.low
    action_high = env.action_space.high
    assert len(obs_shape) == 1
    assert len(action_shape) == 1
    max_action = np.abs(np.concatenate((action_high, action_low))).max().item()
    net = utils.GaussianMlp(obs_shape=obs_shape, action_shape=action_shape, hidden_sizes=hidden_sizes,
                            max_action=max_action, min_variance=min_variance, max_variance=max_variance)
    return utils.GaussianPolicy(net=net, obs_shape=obs_shape, action_shape=action_shape, action_low=action_low,
                                action_high=action_high)


def create_critic(env, hidden_sizes, discount_factor, learning_rate, num_atoms, cost_min, cost_max, risk_type,
                  risk_init_coef, risk_end_coef, risk_warmup_steps, risk_grow_steps, risk_temperature,
                  risk_margin):
    obs_shape = env.observation_space.shape
    assert len(obs_shape) == 1

    reward_layer_sizes = [obs_shape[0]] + hidden_sizes + [1]
    reward_net = utils.create_mlp(reward_layer_sizes)
    reward_optimizer = torch.optim.Adam(reward_net.parameters(), lr=learning_rate)

    cost_layer_sizes = [obs_shape[0]] + hidden_sizes + [num_atoms]
    cost_net = utils.create_mlp(cost_layer_sizes)
    cost_optimizer = torch.optim.Adam(cost_net.parameters(), lr=learning_rate)

    risk_measure = risk.ZeroRisk()
    if risk_type == 'mean':
        risk_measure = risk.MeanRisk(discount_factor=discount_factor)
    if risk_type == 'exp':
        risk_measure = risk.ExpUtilityRisk(discount_factor=discount_factor, temperature=risk_temperature)
    if risk_type == 'var':
        risk_measure = risk.ClippedVarRisk(discount_factor=discount_factor, margin=risk_margin)
    risk_schedule = risk.LinearSchedule(init_value=risk_init_coef, end_value=risk_end_coef,
                                        num_warmup_steps=risk_warmup_steps, num_grow_steps=risk_grow_steps)

    return CTD0(reward_net=reward_net, cost_net=cost_net, reward_optimizer=reward_optimizer,
                cost_optimizer=cost_optimizer, discount_factor=discount_factor, obs_shape=obs_shape,
                num_atoms=num_atoms, cost_min=cost_min, cost_max=cost_max, risk_schedule=risk_schedule,
                risk_measure=risk_measure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='Safexp-PointGoal0-v0')
    # Actor Hyperparameters
    parser.add_argument('--actor_hidden_sizes', nargs='*', type=int, default=[256, 256])
    parser.add_argument('--actor_learning_rate', type=float, default=0.0003)
    parser.add_argument('--min_variance', type=float, default=1e-6)
    parser.add_argument('--max_variance', type=float, default=float('inf'))
    # Critic Hyperparameters
    parser.add_argument('--critic_hidden_sizes', nargs='*', type=int, default=[256, 256])
    parser.add_argument('--critic_learning_rate', type=float, default=0.001)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--num_atoms', type=int, default=51)
    parser.add_argument('--cost_min', type=float, default=0.)
    parser.add_argument('--cost_max', type=float, default=200.)
    # Risk Hyperparameters
    parser.add_argument('--risk_type', choices=('zero', 'mean', 'exp', 'var'), default='zero')
    parser.add_argument('--risk_init_coef', type=float, default=0.)
    parser.add_argument('--risk_end_coef', type=float, default=1.)
    parser.add_argument('--risk_warmup_steps', type=int, default=30)
    parser.add_argument('--risk_grow_steps', type=int, default=90)
    parser.add_argument('--risk_temperature', type=float, default=0.2)
    parser.add_argument('--risk_margin', type=float, default=1.)
    # CPPO Hyperparameters
    parser.add_argument('--num_train_iters', type=int, default=350)
    parser.add_argument('--steps_per_iter', type=int, default=30000)
    parser.add_argument('--num_actor_epochs', type=int, default=80)
    parser.add_argument('--num_critic_epochs', type=int, default=80)
    parser.add_argument('--cost_threshold', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=30000)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.)
    parser.add_argument('--kl_coef', type=float, default=0.)
    parser.add_argument('--target_kl', type=float, default=0.012)
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    parser.add_argument('--save_interval', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--run_dir')
    parser.add_argument('--add_threshold_obs', default=False, action='store_true')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    if args.run_dir is not None:
        os.makedirs(args.run_dir, exist_ok=True)
        args_path = os.path.join(args.run_dir, 'args.json')
        with open(args_path, 'w') as f:
            json.dump(args.__dict__, f)

    env = gym.make(args.env_name)
    if not isinstance(env, Engine):
        print('Non safety_gym environment')
        time.sleep(1)
        env = utils.SafetyWrapper(env)
    env = wrappers.TimeLimit(env=env, max_episode_steps=args.max_episode_steps)
    if args.add_threshold_obs:
        env = utils.CostThresholdWrapper(env=env, cost_threshold=args.cost_threshold)

    if args.seed is not None:
        env.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    actor = create_actor(env=env, hidden_sizes=args.actor_hidden_sizes, min_variance=args.min_variance,
                         max_variance=args.max_variance)
    actor_optimizer = torch.optim.Adam(actor.net.parameters(), lr=args.actor_learning_rate)
    critic = create_critic(env=env, hidden_sizes=args.critic_hidden_sizes, discount_factor=args.discount_factor,
                           learning_rate=args.critic_learning_rate, num_atoms=args.num_atoms, cost_min=args.cost_min,
                           cost_max=args.cost_max, risk_type=args.risk_type, risk_init_coef=args.risk_init_coef,
                           risk_end_coef=args.risk_end_coef, risk_warmup_steps=args.risk_warmup_steps,
                           risk_grow_steps=args.risk_grow_steps, risk_temperature=args.risk_temperature,
                           risk_margin=args.risk_margin)
    hp = CPPOHyperparameters(num_train_iters=args.num_train_iters, steps_per_iter=args.steps_per_iter,
                             save_interval=args.save_interval, run_dir=args.run_dir, cost_threshold=args.cost_threshold,
                             batch_size=args.batch_size, num_actor_epochs=args.num_actor_epochs,
                             num_critic_epochs=args.num_critic_epochs, clip_epsilon=args.clip_epsilon,
                             entropy_coef=args.entropy_coef, kl_coef=args.kl_coef, target_kl=args.target_kl)

    cppo = CPPO(env=env, actor=actor, critic=critic, actor_optimizer=actor_optimizer, hp=hp)
    print(cppo)
    cppo.train()


if __name__ == '__main__':
    main()
