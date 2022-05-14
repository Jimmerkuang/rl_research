# -*- coding: utf-8 -*-
# Time   : 2022/5/11 20:21
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


class PG(object):

    def __init__(self):
        super(PG, self).__init__()
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
        return action.item(), action_prob[:, action].item()

    def get_state_value(self, state):
        state = torch.from_numpy(state).float().view(-1, num_state)
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
            for t in e:
                dis_reward = gama * dis_reward + t.reward
                dis_rewards.append(dis_reward)
        dis_rewards = torch.from_numpy(np.array(dis_rewards[::-1])).float().view(-1, 1)
        return state, action, dis_rewards, old_action_prob

    def update(self):
        state, action, dis_rewards, old_action_prob = self.prepare_training_data_from_buffer()
        print("The agent is updating by policy gradient....")
        action_prob = self.actor(state)
        action_log_prob = torch.vstack([Categorical(a_p).log_prob(a) for a, a_p in zip(action, action_prob)])
        state_value = self.critic(state)
        advantage = (dis_rewards - state_value).detach()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        action_loss = (-action_log_prob * advantage).sum() / len(self.buffer)
        value_loss = F.mse_loss(state_value, dis_rewards)
        action_loss.backward()
        value_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.buffer = []


def discount_and_normalize_rewards(ep_buffer):
    dis_rewards, dis_reward = [], 0
    for r in ep_buffer['reward'][::-1]:
        dis_reward = gama * dis_reward + r
        dis_rewards.append(dis_reward)
    dis_rewards = torch.tensor(dis_rewards[::-1])
    dis_rewards = (dis_rewards - dis_rewards.mean(dim=0)) / (dis_rewards.std(dim=0))
    ep_buffer['discount_reward'] = dis_rewards


def stack_episode_buffer(ep_buffer):
    ep_buffer['state_value'] = torch.hstack(ep_buffer['state_value'])
    ep_buffer['action_log_prob'] = torch.hstack(ep_buffer['action_log_prob'])
    # torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)


def main():
    agent = PG()
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
                # discount_and_normalize_rewards(episode_buffer)
                # stack_episode_buffer(episode_buffer)
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
