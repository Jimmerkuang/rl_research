# -*- coding: utf-8 -*-
# Time   : 2022/5/10 20:48
# Author : kfu


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
from itertools import count
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
from typing import List
from mountain_car import MountainCarEnv


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.action = nn.Linear(128, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action = F.softmax(self.action(x), dim=1)
        return action


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.state_value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO(object):
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10

    def __init__(self):
        super(PPO, self).__init__()
        self.actor = Actor()
        self.critic = Critic()
        self.buffer = []
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.003)

    def select_action(self, state):
        state = torch.from_numpy(state).float().view(-1, num_state)
        with torch.no_grad():
            action_prob = self.actor(state)
        c = Categorical(action_prob)
        action = c.sample()
        action = action.item()
        return action, action_prob[:, action].item()

    def get_state_value(self, state):
        state = torch.from_numpy(state).float().view(-1, num_state)
        with torch.no_grad():
            value = self.critic(state)
        return value

    def store_episode_buffer(self, ep_buffer: List):
        self.buffer.append(ep_buffer)

    def prepare_training_data_from_buffer(self):
        state = torch.from_numpy(np.array([t.state for e in self.buffer for t in e])).float()
        action = torch.from_numpy(np.array([t.action for e in self.buffer for t in e])).long().view(-1, 1)
        old_action_prob = torch.from_numpy(np.array([t.action_prob for e in self.buffer for t in e])).float().view(-1, 1)
        dis_rewards = []
        for e in self.buffer:
            dis_reward = 0
            for t in e[::-1]:
                dis_reward = gama * dis_reward + t.reward
                dis_rewards.append(dis_reward)
        dis_rewards = torch.from_numpy(np.array(dis_rewards[::-1])).float().view(-1, 1)
        return state, action, dis_rewards, old_action_prob

    def update(self):
        state, action, dis_rewards, old_action_prob = self.prepare_training_data_from_buffer()
        print("The agent is updating by PPO...")
        for _ in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(state.shape[0])), 256, False):
                state_value = self.critic(state[index])
                advantage = (dis_rewards[index] - state_value).detach()
                action_prob = self.actor(state[index]).gather(1, action[index])
                important_weight = action_prob / old_action_prob[index]
                surr1 = important_weight * advantage
                surr2 = torch.clamp(important_weight, 1 - self.clip_param, 1 + self.clip_param) * advantage
                action_loss = -torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                value_loss = F.mse_loss(dis_rewards[index], state_value)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
        print('Update done...')
        self.buffer = []


def main():
    agent = PPO()
    durations = []
    for i_episode in tqdm(range(num_episode)):
        state = env.reset()
        episode_buffer = []
        for t in count():
            action, action_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            if render:
                env.render()
            trans = Transition(state, action, action_prob, reward, next_state)
            episode_buffer.append(trans)
            state = next_state
            if done:
                agent.store_episode_buffer(episode_buffer)
                print(f'agent used {t} steps in episode {i_episode}.')
                durations.append(t)
                if len(agent.buffer) >= batch_size:
                    agent.update()
                break
    plt.plot(range(len(durations)), durations)
    plt.show()


# Parameters
gama = 0.99
batch_size = 1
num_episode = 500
render = False
seed = 1

env = MountainCarEnv()
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
# torch.manual_seed(seed)
# env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'action_prob', 'reward', 'next_state'])


if __name__ == '__main__':
    main()
