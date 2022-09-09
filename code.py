#!/usr/bin/env python3

import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import pprint
import highway_env
from highway_env.road.regulation import RegulatedRoad

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
    
    def push(self, *args):
        # Save a Transition
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# class DQN(nn.Module):
#     def __init__(self, h, w, outputs):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm2d(32)
#         self.flatten = nn.Flatten()

#         def conv2d_size_out(size, kernel_size = 5, stride = 2):
#             return (size - (kernel_size - 1) - 1) // stride + 1
        
#         convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
#         convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
#         linear_input_size = convw*convh*32
#         self.fc = nn.Linear(2*2*32,outputs)
    
#     def forward(self, x):
#         x = x.to(device)
#         # print(x.shape)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         # print(x.shape)
#         x = self.flatten(x)
#         # print(x.shape)

#         return self.fc(x.view(x.size(0),-1))

class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8*8*256,64)
        self.fc2 = nn.Linear(64,outputs)
    
    def forward(self, x):
        x = x
        x = self.dropout(self.maxpool(self.relu(self.conv1(x))))
        x = self.dropout(self.maxpool(self.relu(self.conv2(x))))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


def get_screen():
    screen = env.render(mode='rgb_array')
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    screen = screen.permute(2,0,1)
    # print(screen.size())
    return resize(screen).unsqueeze(0)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1. * steps_done/EPS_DECAY)
    steps_done = steps_done + 1
    if sample > eps_threshold:
        with torch.no_grad():
            # print('in')
            # state = state.permute(2,0,1)
            # state = state.type(dtype=torch.FloatTensor) 
            # print(state.type())
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # print(random.randrange(n_actions))
        # return torch.Tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        return torch.Tensor([[random.randrange(n_actions)]])

def plot_durations():
    plt.figure(2)
    plt.clf()
    # durations_t = torch.Tensor(episode_durations, dtype=torch.float)
    agg_reward = torch.Tensor(agg_reward_hist)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(agg_reward.numpy())

    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99),means))
    #     plt.plot(means.numpy())

    if len(agg_reward) >= 100:
        means = agg_reward.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99),means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)

def optimize_model():
    if len(memory) < MIN_REPLAY_MEMORY_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # non_final_mask = torch.Tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_mask = torch.Tensor(tuple(map(lambda s: s is not None, batch.next_state))).type(dtype=torch.BoolTensor)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    # print(state_batch)
    # print('Done')
    action_batch = torch.cat(batch.action)
    action_batch = action_batch.type(dtype=torch.LongTensor)
    # print(action_batch.type())
    reward_batch = torch.cat(batch.reward)

    # state_action_values = policy_net(state_batch).gather(1, action_batch)
    state_action_values = policy_net(state_batch)
    # print(state_action_values)
    state_action_values = state_action_values.gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    loss_history.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

env = gym.make('intersection-v0')
# env = gym.make("CartPole-v1")
env.config["duration"] = 20
env.config["arrived_reward"] = 10
env.config["collision_reward"] = -10

resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])

# plt.imshow(screen.cpu().numpy(),interpolation='none')
# plt.show()

plt.ion()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',('state','action','next_state','reward'))

# BATCH_SIZE = 128
# GAMMA = 0.999
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 2000
# TARGET_UPDATE = 10

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
MIN_REPLAY_MEMORY_SIZE = 1000
AGGREGATE_STATS_EVERY = 50

n_actions = env.action_space.n

policy_net = DQN(n_actions)
target_net = DQN(n_actions)
# policy_net.load_state_dict(torch.load('model.pth'))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

episode_durations = []

num_episodes = 10000
agg_reward_hist = []

for i_episode in range(num_episodes):

    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    loss_history = []
    reward_history = []
    episode_reward = 0

    for t in count():

        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        # reward_history.append(reward)
        episode_reward += reward
        reward = torch.tensor([reward])

        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
        
        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()
        if done:
            # episode_durations.append(t + 1)
            plot_durations()
            break
    agg_reward_hist.append(episode_reward)
    print('Episode', i_episode, 'Loss', np.mean(loss_history), 'Reward', episode_reward)
    
    
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    if np.max(agg_reward_hist) < episode_reward:
        torch.save(policy_net.state_dict(),'model.pth')
    

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()

# torch.save(policy_net.state_dict(),'model.pth')

