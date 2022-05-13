# -*- coding: utf-8 -*-
# Time   : 2022/5/11 20:21
# Author : kfu


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
from typing import Dict
from itertools import count
from mountain_car import MountainCarEnv


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(num_state, 128)
        self.action = nn.Linear(128, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action = F.softmax(self.action(x), dim=0)
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
    batch_size = 1

    def __init__(self):
        super(PG, self).__init__()
        self.actor = Actor()
        self.critic = Critic()
        self.buffer = []
        self.total_rewards = []
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.003)

    def select_action(self, state):
        action_prob = self.actor(state)
        c = Categorical(action_prob)
        action = c.sample()
        action_log_prob = c.log_prob(action)
        return action.item(), action_log_prob

    def get_state_value(self, state):
        value = self.critic(state)
        return value

    def store_episode_buffer(self, ep_buffer: Dict):
        self.buffer.append(ep_buffer)

    def update(self):
        # print("The agent is updating....")
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        for b in self.buffer:
            action_loss = (-b['action_log_prob'] * (b['discount_reward'] - b['state_value'])).mean()
            value_loss = ((b['state_value'] - b['discount_reward']) ** 2).mean()
            loss = action_loss + value_loss
            loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.buffer = []


def create_empty_episode_buffer():
    episode_buffer = {
        'state': [],
        'state_value': [],
        'action': [],
        'action_log_prob': [],
        'reward': [],
        'discount_reward': None,
        }
    return episode_buffer


def store_transition(ep_buffer: Dict, s: torch.Tensor, s_v: torch.Tensor, a: int, a_lp: torch.Tensor, r: float):
    ep_buffer['state'].append(s)
    ep_buffer['state_value'].append(s_v)
    ep_buffer['action'].append(a)
    ep_buffer['action_log_prob'].append(a_lp)
    ep_buffer['reward'].append(r)


def discount_and_normalize_rewards(ep_buffer: Dict):
    dis_rewards, dis_reward = [], 0
    for r in ep_buffer['reward'][::-1]:
        dis_reward = gama * dis_reward + r
        dis_rewards.append(dis_reward)
    dis_rewards = torch.tensor(dis_rewards[::-1])
    dis_rewards = (dis_rewards - dis_rewards.mean(dim=0)) / (dis_rewards.std(dim=0))
    ep_buffer['discount_reward'] = dis_rewards


def stack_episode_buffer(ep_buffer: Dict):
    ep_buffer['state_value'] = torch.hstack(ep_buffer['state_value'])
    ep_buffer['action_log_prob'] = torch.hstack(ep_buffer['action_log_prob'])


def main():
    agent = PG()
    durations = []
    for i_episode in tqdm(range(10)):
        state = env.reset()
        if render:
            env.render()
        episode_buffer = create_empty_episode_buffer()
        for t in count():
            action, action_log_prob = agent.select_action(state)
            state_value = agent.get_state_value(state)
            next_state, reward, done, _ = env.step(action)
            store_transition(episode_buffer, state, state_value, action, action_log_prob, reward)
            state = next_state
            if render:
                env.render()
            if done:
                discount_and_normalize_rewards(episode_buffer)
                stack_episode_buffer(episode_buffer)
                agent.store_episode_buffer(episode_buffer)
                if len(agent.buffer) >= agent.batch_size:
                    agent.update()
                print(f'agent used {t} steps in episode {i_episode}.')
                durations.append(t)
                break
    plt.plot(range(len(durations)), durations)
    plt.show()


# Parameters
gama = 0.99
render = False
seed = 1

env = MountainCarEnv()
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
torch.manual_seed(seed)
env.seed(seed)


if __name__ == '__main__':
    main()
