import gym
import torch
import numpy as np
from torch import nn
import random
import torch.nn.functional as F
import collections
from torch.optim.lr_scheduler import StepLR
from collections import deque
import copy
from matplotlib import pyplot as plt
from numpy import savetxt

"""
Implementation of DQN for gym environments with discrete action space.
source: 
"""

def test_model(model, env,state_flattened_size):
    _state = env.reset()
    state = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, state_flattened_size)
    done = False
    rewards = []
    while not done:
        # DQN
        qval = model(state)
        qval_ = qval.data.numpy()
        action = np.argmax(qval_)
        _state, reward, done, _ = env.step(action)
        state = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, state_flattened_size)
        rewards.append(reward)
        print("Request:", len(rewards), "Action:", action, "Reward:", reward)
    print("Reward sum:", sum(rewards))
    return rewards

def save_plot_and_csv_train(total_rewards):
    x = np.arange(len(total_rewards))
    y = total_rewards
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Reward", fontsize=22)
    plt.plot(x, y, c='black')
    plt.savefig('dqn_train.png')
    # save to csv file
    savetxt('dqn_train.csv', total_rewards, delimiter=',')

def save_plot_and_csv_test(total_rewards):
    x = np.arange(len(total_rewards))
    y = total_rewards
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Reward", fontsize=22)
    plt.plot(x, y, c='black')
    plt.savefig('dqn_test.png')
    # save to csv file
    savetxt('dqn_test.csv', total_rewards, delimiter=',')


def dqn_agent(gamma = 0.9, epsilon = 0.5, learning_rate = 1e-3,state_flattened_size = 845, epochs = 4000,mem_size = 50000,
    batch_size = 256,sync_freq = 16,l1 = 845, l2 = 1500, l3 = 700,l4 = 200, l5 = 5, env=""):
    """
    :param gamma: reward discount factor
    :param epsilon: probability to take a random action during training
    :param learning_rate: learning rate for the Q-Network
    :param batch_size: see above
    :param env_name: name of the gym environment
    :return: 
    """
    

    env= env

    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4),
        torch.nn.ReLU(),
        torch.nn.Linear(l4, l5)
    )

    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    total_reward_list = []

    n_action = env.action_space.n
    
    replay = deque(maxlen=mem_size)
    
    for i in range(epochs):
        print("Starting training, epoch:", i)
        cnt = 0
        total_reward = 0
        _state = env.get_state()
        state1 = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, state_flattened_size)
        done = False
        env.reset()
        print("TRAIN AGENT")
        while not done:
            print("Step:", cnt + 1)
            cnt += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            
            if (random.random() < epsilon):
                action_ = np.random.randint(0, n_action - 1)
            else:
                action_ = np.argmax(qval_)

            state, reward, done, _ = env.step(action_)
            state2 = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1, state_flattened_size)

            exp = (state1, action_, reward, state2, done)

            replay.append(exp)
            state1 = state2

            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model2(state2_batch)

                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                print(i, loss.item())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

                if cnt % sync_freq == 0:
                    model2.load_state_dict(model.state_dict())
            
            print("Reward:",reward)
            total_reward += reward

        total_reward_list.append(total_reward)
        save_plot_and_csv_train(total_reward_list)

        print("Episode reward:", total_reward)

        if epsilon > 0.01:
            epsilon -= (1 / epochs)

   
    
    #GUARDAR total_reward_list do TRAIN
    torch.save(model.state_dict(), 'dqn.pt')
    print("TEST AGENT")
    model_test=model
    model_test.load_state_dict(torch.load("dqn.pt"))
    test_rewards=test_model(model_test,env,state_flattened_size)
    save_plot_and_csv_test(test_rewards)
    
    #***************************PLOT TRAIN AND TEST*******************************************
"""
    epochs_array=[]
    rewards_array=[]
    for epochs_n, epochs_reward in enumerate(total_reward_list):
        epochs_array.append(epochs_n)
        rewards_array.append(epochs_reward)

    plt.figure(figsize=(10, 7))
    plt.plot(epochs_array,rewards_array,linewidth=2.0)
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Reward", fontsize=22)
    plt.savefig('dqn_train.png')
    # save to csv file
    savetxt('dqn_train.csv', total_reward_list, delimiter=',')
    
    steps_array=[]
    rewards_array=[]
    for epochs_n, epochs_reward in enumerate(test_rewards):
        steps_array.append(epochs_n)
        rewards_array.append(epochs_reward)

    plt.figure(figsize=(10, 7))
    plt.plot(steps_array,rewards_array,linewidth=2.0)
    plt.xlabel("Steps", fontsize=22)
    plt.ylabel("Reward", fontsize=22)
    plt.savefig('dqn_test.png')
    # save to csv file
    savetxt('dqn_test.csv', test_rewards, delimiter=',')
"""
    
     