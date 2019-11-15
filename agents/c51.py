from collections import namedtuple
import copy
import random
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import ai_safety_gridworlds

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


Transition = namedtuple('Transition', ('obs', 'action', 'reward', 'next_obs', 'done'))


class ReplayBuffer:
    def __init__(self, max_size):
        assert max_size > 0
        self.max_size = max_size
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, x):
        if len(self.memory) < self.max_size:
            self.memory.append(None)
        self.memory[self.position] = x
        self.position = (self.position + 1) % self.max_size

    def sample(self, size):
        assert len(self.memory) >= size
        return random.sample(self.memory, size)


class LinearSchedule:
    def __init__(self, init_epsilon, decay_period, num_warmup_steps, min_epsilon):
        self.init_epsilon = init_epsilon
        self.decay_period = decay_period
        self.num_warmup_steps = num_warmup_steps
        self.min_epsilon = min_epsilon

    def get(self, step):
        if step <= self.num_warmup_steps:
            return self.init_epsilon
        if step >= self.decay_period + self.num_warmup_steps:
            return self.min_epsilon
        decay_steps = step - self.num_warmup_steps
        return self.init_epsilon - decay_steps * (self.init_epsilon - self.min_epsilon) / self.decay_period


class BellmanOp(nn.Module):
    def __init__(self, v_min, v_max, num_atoms):
        super().__init__()
        assert v_max > v_min
        assert num_atoms > 1
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.atom_delta = (v_max - v_min) / (num_atoms - 1)
        self.atom_values = torch.tensor([v_min + self.atom_delta * i for i in range(self.num_atoms)])


    def forward(self, reward, probs):
        bs = probs.shape[0]
        assert reward.shape == (bs,)
        assert probs.shape == (bs, self.num_atoms)
        # assert torch.all(torch.isclose(probs.sum(dim=1), torch.ones((bs,))))

        atom_values = self.atom_values.unsqueeze(0).expand((bs, self.num_atoms))
        reward = reward.unsqueeze(-1).expand((bs, self.num_atoms))
        new_atom_values = atom_values + reward
        new_atom_values = torch.clamp(new_atom_values, min=self.v_min, max=self.v_max)

        new_atom_indices = (new_atom_values - self.v_min) / self.atom_delta
        lower_index = torch.floor(new_atom_indices)
        upper_index = torch.ceil(new_atom_indices)
        lower_coef = upper_index - new_atom_indices
        upper_coef = new_atom_indices - lower_index

        projection_matrix = torch.zeros((bs, self.num_atoms, self.num_atoms))
        for i in range(bs):
            for j in range(self.num_atoms):
                l = int(lower_index[i, j].item())
                l_coef = lower_coef[i, j]
                u = int(upper_index[i, j].item())
                u_coef = upper_coef[i, j]
                projection_matrix[i, l, j] = l_coef
                projection_matrix[i, u, j] = u_coef
                if l == u:
                    assert l_coef == 0
                    assert u_coef == 0
                    projection_matrix[i, l, j] = 1.

        probs = probs.unsqueeze(-1)
        assert probs.shape == (bs, self.num_atoms, 1)
        new_probs = torch.bmm(projection_matrix, probs).squeeze(-1)
        assert new_probs.shape == (bs, self.num_atoms)
        # assert torch.all(torch.isclose(new_probs.sum(dim=1), torch.ones((bs,))))
        return new_probs


class C51MLP(nn.Module):
    def __init__(self, obs_size, hidden_sizes, num_actions, num_atoms, v_min, v_max):
        super().__init__()
        assert v_max > v_min
        assert num_atoms > 1
        layers = []
        input_size = obs_size
        for output_size in hidden_sizes:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
            input_size = output_size
        layers.append(nn.Linear(input_size, num_actions * num_atoms))
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.atom_delta = (v_max - v_min) / (num_atoms - 1)
        self.atom_values = torch.tensor([v_min + self.atom_delta * i for i in range(num_atoms)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        bs = x.shape[0]
        assert x.shape == (bs, self.obs_size)
        x = self.net(x)
        assert x.shape == (bs, self.num_actions * self.num_atoms)
        logits = x.view(bs, self.num_actions, self.num_atoms)
        logits = torch.clamp(logits, min=0, max=20)
        probs = F.softmax(logits, dim=-1)
        assert probs.shape == (bs, self.num_actions, self.num_atoms)
        action_values = (self.atom_values * probs).sum(dim=-1)
        assert action_values.shape == (bs, self.num_actions)
        return probs, action_values


class DistributionCrossEntropyLoss(nn.Module):
    def forward(self, input, target):
        assert input.shape == target.shape
        assert len(input.shape) == 2
        return -(torch.log(input) * target).sum(dim=-1).mean()


class C51:
    def __init__(self, env, net, optimizer, loss_fn, epsilon_schedule, replay_buffer_size, num_warmup_steps,
                 num_train_steps, batch_size, max_grad_norm, target_update_interval, log_interval, eval_interval,
                 num_eval_episodes, v_min, v_max, num_atoms):
        self.env = env
        self.net = net
        self.target_net = copy.deepcopy(net)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epsilon_schedule = epsilon_schedule
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.num_warmup_steps = num_warmup_steps
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.target_update_interval = target_update_interval
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.num_actions = env.action_space.n
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.bellman_op = BellmanOp(v_min=v_min, v_max=v_max, num_atoms=num_atoms)
        self.zero_atom_index = 0 # TODO
        self._curr_return = 0
        self._training_returns = []
        self._training_losses = []

    def sample_action(self, obs, step):
        if step <= self.num_warmup_steps:
            return self.env.action_space.sample()
        epsilon = self.epsilon_schedule.get(step)
        if random.random() <= epsilon:
            return self.env.action_space.sample()
        return self.greedy_action(obs)

    def greedy_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        if len(obs.shape) == len(self.env.observation_space.shape):
            obs = obs.unsqueeze(0)
        assert obs.shape[1:] == self.env.observation_space.shape
        probs, action_values = self.net(obs)
        # print('probs:', probs)
        # print('action_values:', action_values)
        return action_values.squeeze().argmax().item()

    def train(self):
        obs = self.env.reset()
        for step in range(1, self.num_train_steps):
            action = self.sample_action(obs=obs, step=step)
            next_obs, reward, done, info = self.env.step(action)
            transition = Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
            self.record_transition(transition)
            obs = self.env.reset() if done else next_obs
            self.update_net(step)
            self.update_target_net(step)
            self.log_metrics(step)
            # obs = self.eval(step, obs)

    # def eval(self, step, obs):
    #     if step <= self.num_warmup_steps:
    #         return obs
    #     if (step - self.num_warmup_steps) % self.eval_interval != 0:
    #         return obs
    #     episode_returns = []
    #     for _ in range(self.num_eval_episodes):
    #         obs = self.env.reset()
    #         done = False
    #         curr_return = 0
    #         while not done:
    #             action = self.greedy_action(obs)
    #             obs, reward, done, _ = self.env.step(action)
    #             curr_return += reward
    #         episode_returns.append(curr_return)
    #     avg_return = sum(episode_returns) / len(episode_returns)
    #     print('* [Step {}] eval_ret: {:.03f}'.format(step, avg_return))
    #     return self.env.reset()
    #
    def log_metrics(self, step):
        if step < self.num_warmup_steps:
            return
        if step == self.num_warmup_steps:
            self._train_start = time.time()
            return
        if (step - self.num_warmup_steps) % self.log_interval == 0:
            avg_train_ret = sum(self._training_returns) / len(self._training_returns)
            avg_train_loss = sum(self._training_losses) / len(self._training_losses)
            train_elapsed = time.time() - self._train_start
            steps_per_sec = self.log_interval / train_elapsed
            self._training_returns.clear()
            self._training_losses.clear()
            self._train_start = time.time()
            print('[Step {}] train_ret: {:.03f}, train_loss: {:.03f}, 1k_steps_per_sec: {:.02f}'.format(
                step, avg_train_ret, avg_train_loss, steps_per_sec / 1000))

    def record_transition(self, transition):
        self.replay_buffer.push(transition)
        self._curr_return += transition.reward
        if transition.done:
            self._training_returns.append(self._curr_return)
            self._curr_return = 0

    def update_net(self, step):
        if step <= self.num_warmup_steps or len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        obs = torch.tensor([x.obs for x in transitions], dtype=torch.float32)
        next_obs = torch.tensor([x.next_obs for x in transitions], dtype=torch.float32)
        rewards = torch.tensor([x.reward for x in transitions], dtype=torch.float32)
        actions = [x.action for x in transitions]
        done_indices = [i for i, x in enumerate(transitions) if x.done]

        target_probs, target_action_values = self.target_net(next_obs)
        assert target_action_values.shape == (self.batch_size, self.env.action_space.n)
        assert target_probs.shape == (self.batch_size, self.num_actions, self.num_atoms)
        greedy_actions = target_action_values.argmax(dim=-1)
        assert greedy_actions.shape == (self.batch_size,)
        greedy_probs = target_probs[range(self.batch_size), greedy_actions]
        assert greedy_probs.shape == (self.batch_size, self.num_atoms)
        greedy_probs[done_indices] *= 0.
        greedy_probs[done_indices, self.zero_atom_index] = 1.
        new_probs = self.bellman_op(reward=rewards, probs=greedy_probs).detach()

        self.optimizer.zero_grad()
        predicted_probs, _ = self.net(obs)
        assert predicted_probs.shape == (self.batch_size, self.num_actions, self.num_atoms)
        predicted_probs = predicted_probs[range(self.batch_size), actions]
        assert predicted_probs.shape == (self.batch_size, self.num_atoms)
        loss = self.loss_fn(input=predicted_probs, target=new_probs)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        self._training_losses.append(loss.item())

    def update_target_net(self, step):
        if step <= self.num_warmup_steps:
            return
        if (step - self.num_warmup_steps) % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())


def test_bellman_op():
    op = BellmanOp(
        v_min = 10,
        v_max = 20,
        num_atoms=6
    )
    dist = torch.tensor([
        #10,  12, 14,  16,  18  20
        [0.2, 0., 0.3, 0.4, 0., 0.1],
        [0.1, 0., 0.5, 0.05, 0.25, 0.1],
    ])
    reward = torch.tensor([
        1.5,
        -0.75
    ])
    out = op(reward=reward, probs=dist)
    print('out:', out)


def main():
    env_name = 'CartPole-v1'
    hidden_sizes = [64, 64]
    num_atoms = 51
    v_min = 0
    v_max = 500
    learning_rate = 0.01
    init_epsilon = 1.
    min_epsilon = 0.1
    num_warmup_steps = 1000
    num_train_steps = 1000000
    decay_period = 10000
    replay_buffer_size = 10000
    batch_size = 128
    max_grad_norm = 5
    target_update_interval = 500
    log_interval = 1000
    eval_interval = 10000
    num_eval_episodes = 10

    env = gym.make(env_name)
    net = C51MLP(
        obs_size=env.observation_space.shape[0],
        hidden_sizes=hidden_sizes,
        num_actions=env.action_space.n,
        num_atoms=num_atoms,
        v_min=v_min,
        v_max=v_max
    )
    print(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = DistributionCrossEntropyLoss()
    epsilon_schedule = LinearSchedule(
        init_epsilon=init_epsilon,
        min_epsilon=min_epsilon,
        num_warmup_steps=num_warmup_steps,
        decay_period=decay_period
    )
    agent = C51(
        env=env,
        net=net,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epsilon_schedule=epsilon_schedule,
        replay_buffer_size=replay_buffer_size,
        num_warmup_steps=num_warmup_steps,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        max_grad_norm=max_grad_norm,
        target_update_interval=target_update_interval,
        log_interval=log_interval,
        eval_interval=eval_interval,
        num_eval_episodes=num_eval_episodes,
        v_min=v_min,
        v_max=v_max,
        num_atoms=num_atoms
    )
    agent.train()




if __name__ == '__main__':
    main()
