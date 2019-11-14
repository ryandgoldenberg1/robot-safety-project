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


class DQN:
    def __init__(self, env, net, optimizer, loss_fn, epsilon_schedule, replay_buffer_size, num_warmup_steps,
                 num_train_steps, batch_size, max_grad_norm, target_update_interval, log_interval, eval_interval,
                 num_eval_episodes):
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
        action_values = self.net(obs).squeeze()
        return action_values.argmax().item()

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
            obs = self.eval(step, obs)

    def eval(self, step, obs):
        if step <= self.num_warmup_steps:
            return obs
        if (step - self.num_warmup_steps) % self.eval_interval != 0:
            return obs
        episode_returns = []
        for _ in range(self.num_eval_episodes):
            obs = self.env.reset()
            done = False
            curr_return = 0
            while not done:
                action = self.greedy_action(obs)
                obs, reward, done, _ = self.env.step(action)
                curr_return += reward
            episode_returns.append(curr_return)
        avg_return = sum(episode_returns) / len(episode_returns)
        print('* [Step {}] eval_ret: {:.03f}'.format(step, avg_return))
        return self.env.reset()

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
        done_mask = torch.tensor([0 if x.done else 1 for x in transitions], dtype=torch.float32)
        actions = [x.action for x in transitions]

        target_values = self.target_net(next_obs).max(dim=1).values
        assert target_values.shape == (self.batch_size,)
        target_values = (rewards + done_mask * target_values).detach()

        self.optimizer.zero_grad()
        predicted_values = self.net(obs)[range(self.batch_size), actions]
        assert predicted_values.shape == (self.batch_size,)
        loss = self.loss_fn(predicted_values, target_values)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        self._training_losses.append(loss.item())

    def update_target_net(self, step):
        if step <= self.num_warmup_steps:
            return
        if (step - self.num_warmup_steps) % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())



# Tests

def test_schedule():
    import matplotlib.pyplot as plt
    schedule = LinearSchedule(
        init_epsilon=0.8,
        decay_period=10000,
        num_warmup_steps=1000,
        min_epsilon=0.1
    )
    x = list(range(1, 20000+1))
    y = [schedule.get(z) for z in x]
    plt.plot(x, y)
    plt.show()


def test_replay_buffer():
    buffer = ReplayBuffer(max_size=10)
    for i in range(20):
        buffer.push(i)
        print(i)
        print(buffer.__dict__)
    print(buffer.sample(5))


def test_env_speed():
    import time
    env = gym.make('CartPole-v1')

    start = time.time()
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        if done:
            env.reset()
    raw_elapsed = time.time() - start

    dqn = DQN(
        env=env,
        net=nn.Linear(4,2),
        optimizer=None,
        loss_fn=None,
        epsilon_schedule=LinearSchedule(init_epsilon=1., min_epsilon=0.1, num_warmup_steps=500, decay_period=500),
        replay_buffer_size=500,
        num_warmup_steps=500,
        num_train_steps=1000,
        batch_size=1
    )
    start = time.time()
    dqn.train()
    dqn_elapsed = time.time() - start

    print('raw:', raw_elapsed)
    print('dqn:', dqn_elapsed)


def main():
    env_name = 'DistributionalShift-v0'
    learning_rate = 0.001
    init_epsilon = 0.5
    min_epsilon = 0.1
    num_warmup_steps = 1000
    decay_period = 100000
    replay_buffer_size = 100000
    num_train_steps = 1000000
    batch_size = 256
    max_grad_norm = 2
    target_update_interval = 1000
    log_interval = 1000
    eval_interval = 10000
    num_eval_episodes = 10
    reward_scale = 1.

    env = gym.make(env_name, reward_scale=reward_scale)
    # Input (bs, 5, 7, 9)

    class ShapeCheck(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape

        def forward(self, x):
            bs = x.shape[0]
            assert x.shape == (bs, *self.shape), 'Exp: {}, Actual: {}'.format(self.shape, x.shape)
            return x

    net = nn.Sequential(
        nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3, stride=1, padding=1), # (bs, 8, 7, 9)
        # ShapeCheck(shape=(8, 7, 9)),
        nn.ReLU(),
        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0), # (bs, 16, 3, 5)
        # ShapeCheck(shape=(16, 3, 5)),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0), # (bs, 32, 1, 3)
        # ShapeCheck(shape=(32, 1, 3)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(96, 32),
        nn.ReLU(),
        nn.Linear(32, 4)
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    epsilon_schedule = LinearSchedule(
        init_epsilon=init_epsilon,
        min_epsilon=min_epsilon,
        num_warmup_steps=num_warmup_steps,
        decay_period=decay_period
    )
    agent = DQN(
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
        num_eval_episodes=num_eval_episodes
    )
    agent.train()


if __name__ == '__main__':
    main()
