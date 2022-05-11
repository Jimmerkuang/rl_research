# -*- coding: utf-8 -*-
# Time   : 2022/5/11 20:21
# Author : kfu


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count
import numpy as np
import gym
from mountain_car import MountainCarEnv


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(2, 128)
        self.action = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action = F.softmax(self.action(x))
        return action


def main():

    # Parameters
    num_episode = 5000
    batch_size = 1
    learning_rate = 0.01
    gamma = 0.99
    render = False

    env = MountainCarEnv()
    policy_net = PolicyNet()
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)

    # Batch History
    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0

    for e in range(num_episode):

        state = env.reset()
        state = torch.from_numpy(state).float()
        state = Variable(state)
        if e > 1:
            env.render()

        for t in count():

            probs = policy_net(state)
            m = Bernoulli(probs)
            action = m.sample()

            action = action.data.numpy().astype(int)[0]
            next_state, reward, done, _ = env.step(action)
            if e > 1:
                env.render()

            # To mark boundarys between episodes
            if done:
                reward = 0

            state_pool.append(state)
            action_pool.append(float(action))
            reward_pool.append(reward)

            state = next_state
            state = torch.from_numpy(state).float()
            state = Variable(state)

            steps += 1

            if done:
                break

        # Update policy
        if e > 0 and e % batch_size == 0:
            print("The agent is updateing....")
            # Discount reward
            running_add = 0
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * gamma + reward_pool[i]
                    reward_pool[i] = running_add

            # Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            for i in range(steps):
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            # Gradient Desent
            optimizer.zero_grad()

            for i in range(steps):
                state = state_pool[i]
                action = Variable(torch.FloatTensor([action_pool[i]]))
                reward = reward_pool[i]

                probs = policy_net(state)
                m = Bernoulli(probs)
                loss = -m.log_prob(action) * reward  # Negtive score function x reward
                loss.mean().backward() # ?

            optimizer.step()

            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0


if __name__ == '__main__':
    main()
