<!---# Highway-Intersection-Turning-in-Autonomous-Car-Using-Deep-Reinforcement-Learning-PyTorch-
Developed a reinforcement learning pipeline to teach an agent to take turn at highway intersection without colliding with other vehicle using Deep Q-Learning and Epsilon Greedy strategy.-->

## Project Description
The aim of this project is to train an ego vehicle to cross a highway intersection safely avoiding collision with other vehicles approaching the intersection. Deep Q-Network (DQN) is used as the policy network with epsilon-greedy algorithm for selecting actions. Target network is used to predcit the maximum expected future rewards. This rewards and the actual reward received by taking the action is used to compute loss value. This loss value then backpropogated to train the policy network. The ego vehicle is rewarded for reaching the final position after crossing the intersection and penalized for collisions with other vehicles. It is also rewarded for moving at higher speeds.

## Reinforcement Learning Pipeline

![img](https://user-images.githubusercontent.com/90370308/218330577-bc803c33-0d31-48ca-8173-e57932347957.png)

### Agent
The agent is the ego vehcile who is learning how to cross the intersection. It is shown by green box in the above image.

### Environment
The Environment is the highway intersection and all the vehicle approaching it. The ego vehicle is shown by green box and the other vehicle are shown by blue box.

### Action
The ego vehicle has to follow the predefined path. It can take following actions while taking that path:
1. Move fast
2. Move slow
3. Remain stationary

The episode is divided into fixed and equal time interval. At each time interval, vehicle has to decide which action to take. Each action will decide the position of the vehicle along the path at the next time interval. These action are sufficient enough to cross the intersection without colliding with other vehicle.

### Rewards
Rewards associated with each action are as follow:
1. Move fast : 1
2. Move slow : 0
3. Remain stationary : 0
4. Reach goal : 10
5. Collide : -5

### Policy Network

The policy network used here is CNN. It maps the state to the action. The optimal policy network will make the state to action in such a way that ego vehicle can cross the intersection without collision.

## Algorithm

The algorithm used here is Deep-Q Network with epsilon-greedy policy. The pseudocode of the algorithm is as follow:

![Screenshot from 2023-02-12 11-21-57](https://user-images.githubusercontent.com/90370308/218331461-68c2dda7-3a87-4c2a-936f-7573535b4fb4.png)

The flow chart of the training process is as follow:

![690_robot_learning](https://user-images.githubusercontent.com/90370308/218331476-02327a65-f879-492c-9571-59c4c7e7cff2.png)


## Results

1. Behavior of the vehicle before training

![before_training_RL_AdobeExpress](https://user-images.githubusercontent.com/90370308/218331567-6472f87b-8a69-444a-a9e0-d1a5fb815d3f.gif)

2. Behavior of the vehicle after training

![Highway_intersection_crossing_using_Deep_Reinforcement_Learning_AdobeExpress](https://user-images.githubusercontent.com/90370308/218331595-f351b0b2-2d85-4504-bbd0-ba52d0d2d2c6.gif)

## How to run the code?

```
git clone https://github.com/Pradip-Kathiriya/Highway-Intersection-Turning-in-Autonomous-Car-Using-Deep-Reinforcement-Learning-PyTorch-
pip install highway-env
code.py
```



## Requirement
Python 2.0 or above

## License

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Copyright (c) Feb 2023 Pradip Kathiriya
