<!---# Highway-Intersection-Turning-in-Autonomous-Car-Using-Deep-Reinforcement-Learning-PyTorch-
Developed a reinforcement learning pipeline to teach an agent to take turn at highway intersection without colliding with other vehicle using Deep Q-Learning and Epsilon Greedy strategy.-->

## Project Description
The aim of this project is to train an ego vehicle to cross a highway intersection safely avoiding collision with other vehicles approaching the intersection. Deep Q-Network (DQN) is used as the policy network with epsilon-greedy algorithm for selecting actions. Target network is used to predcit the maximum expected future rewards. This rewards and the actual reward received by taking the action is used to compute loss value. This loss value then backpropogated to train the policy network. The ego vehicle is rewarded for reaching the final position after crossing the intersection and penalized for collisions with other vehicles. It is also rewarded for moving at higher speeds.

## Reinforcement Learning Pipeline

![img](https://user-images.githubusercontent.com/90370308/218330577-bc803c33-0d31-48ca-8173-e57932347957.png)

### Agent
The agent is the ego vehcile who is learning how to cross the intersection

### Environment
The Environment is the highway intersection and all the vehicle approaching it.

### Action
The ego vehicle has to follow the predefined path. It can take following actions while taking that path:
1. Move fast
2. Move slow
3. Remain stationary

The episode is divided into fixed and equal time interval. At each time interval, vehicle has to decide which action to take. Each action will decide the position of the vehicle along the path at the next time interval. These action are sufficient enough to cross the intersection without colliding with other vehicle.

### Rewards
Rewards associated with each action are as follow:
1. Move fast - 1
2. Move slow - 0
3. Remain stationary - 0
4. Reach goal - 10
5. Collide - (-5)






## Requirement
Python 2.0 or above

## License

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Copyright (c) Feb 2023 Pradip Kathiriya
