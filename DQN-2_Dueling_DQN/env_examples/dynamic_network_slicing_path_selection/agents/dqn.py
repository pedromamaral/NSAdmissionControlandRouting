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

def test_model(model1,model2, env,state_flattened_size):
    _state = env.reset()
    accepted_slices_test = 0
    total_requests_test = 0
    accepted_test_window = []
    requests_test_window = []
    state = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, state_flattened_size)
    done = False
    rewards = []
    while not done:
        # DQN
        qval1 = model1(state)
        qval1_ = qval1.data.numpy()
 
        qval2=model2(state)
        qval2_=qval2.data.numpy()

        action1 = np.argmax(qval1_)
        action2 = np.argmax(qval2_)
 
        action=action1,action2

        _state, reward, done, accepted_slices_test, total_requests_test, _ = env.step(action)
        state = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, state_flattened_size)
        rewards.append(reward)
        print("Request:", len(rewards), "Action:", action, "Reward:", reward)
    print("Reward sum:", sum(rewards))
    return rewards

def save_plot_and_csv_train(total_rewards, accepted_train_window, requests_train_window):
    x = np.arange(len(total_rewards))
    y = total_rewards
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Reward", fontsize=22)
    plt.plot(x, y, c='black')
    plt.savefig('dqn_train.png')
    # save to csv file
    savetxt('dqn_train.csv', total_rewards, delimiter=',')
    savetxt('accepted_train.csv', accepted_train_window, delimiter=',')
    savetxt('requests_train.csv', requests_train_window, delimiter=',')


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
    batch_size = 256,sync_freq = 16,l1 = 845, l2 = 1500, l3 = 700,l4 = 200, output1 = 2,output2=7, env=""):
    """
    :param gamma: reward discount factor
    :param epsilon: probability to take a random action during training
    :param learning_rate: learning rate for the Q-Network
    :param batch_size: see above
    :param env_name: name of the gym environment
    :return: 
    """
    

    env= env
    total_reward_list = []
    accepted_train_window = []
    requests_train_window = []

#***********************************************************************************
    model1 = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4),
        torch.nn.ReLU(),
        torch.nn.Linear(l4, output1)
    )
    model2 = copy.deepcopy(model1)
    model2.load_state_dict(model1.state_dict())

    loss_fn1 = torch.nn.MSELoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)

    losses1 = []
    n_action1 = output1
    replay1= deque(maxlen=mem_size)
#***********************************************************************************
#***********************************************************************************
    model3=torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4),
        torch.nn.ReLU(),
        torch.nn.Linear(l4, output2)
    )

    model4 = copy.deepcopy(model3)
    model4.load_state_dict(model3.state_dict())

    loss_fn2 = torch.nn.MSELoss()
    optimizer2 = torch.optim.Adam(model3.parameters(), lr=learning_rate)

    losses2 = []
    n_action2 = output2
    replay2= deque(maxlen=mem_size)


    
    #***********************************************************************************
    
    for i in range(epochs):

        print("Starting training, epoch:", i)
        cnt = 0
        accepted_value_train = 0
        requests_value_train = 0
        total_reward = 0
        _state = env.get_state()
        state1 = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, state_flattened_size)
        done = False
        env.reset()
        print("TRAIN AGENT")
        while not done:
            accepted_slices_train = 0
            total_requests_train = 0
            print("Step:", cnt + 1)
            cnt += 1
            qval1 = model1(state1)
            qval1_ = qval1.data.numpy()
            qval2 = model3(state1)
            qval2_ = qval2.data.numpy()
            
            if (random.random() < epsilon):
                action1_=np.random.randint(0, n_action1 - 1)
                action2_=np.random.randint(0, n_action2 - 1)
                action_ = action1_,action2_
            else:
                action1_ = np.argmax(qval1_)
                action2_ = np.argmax(qval2_)
                action_ = action1_,action2_

            state, reward, done, accepted_value_train, requests_value_train, _ = env.step(action_)

            state2 = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1, state_flattened_size)

            exp1 = (state1, action1_, reward, state2, done)
            exp2 = (state1, action2_, reward, state2, done)

            replay1.append(exp1)
            replay2.append(exp2)
            state1 = state2

            if len(replay1) > batch_size:
                minibatch1 = random.sample(replay1, batch_size)
                state1_batch1 = torch.cat([s1 for (s1, a, r, s2, d) in minibatch1])
                action_batch1 = torch.Tensor([a for (s1, a, r, s2, d) in minibatch1])
                reward_batch1 = torch.Tensor([r for (s1, a, r, s2, d) in minibatch1])
                state2_batch1 = torch.cat([s2 for (s1, a, r, s2, d) in minibatch1])
                done_batch1 = torch.Tensor([d for (s1, a, r, s2, d) in minibatch1])
                Q1 = model1(state1_batch1)
                with torch.no_grad():
                    Q2 = model2(state2_batch1)

                Y1 = reward_batch1 + gamma * ((1 - done_batch1) * torch.max(Q2, dim=1)[0])
                X1 = Q1.gather(dim=1, index=action_batch1.long().unsqueeze(dim=1)).squeeze()
                loss1 = loss_fn1(X1, Y1.detach())
                print(i, loss1.item())
                optimizer1.zero_grad()
                loss1.backward()
                losses1.append(loss1.item())
                optimizer1.step()

                if cnt % sync_freq == 0:
                    model2.load_state_dict(model1.state_dict())
            
            if len(replay2) > batch_size:
                minibatch2 = random.sample(replay2, batch_size)
                state1_batch2 = torch.cat([s1 for (s1, a, r, s2, d) in minibatch2])
                action_batch2 = torch.Tensor([a for (s1, a, r, s2, d) in minibatch2])
                reward_batch2 = torch.Tensor([r for (s1, a, r, s2, d) in minibatch2])
                state2_batch2 = torch.cat([s2 for (s1, a, r, s2, d) in minibatch2])
                done_batch2 = torch.Tensor([d for (s1, a, r, s2, d) in minibatch2])
                Q3 = model3(state1_batch2)
                with torch.no_grad():
                    Q4 = model4(state2_batch2)

                Y2 = reward_batch2 + gamma * ((1 - done_batch2) * torch.max(Q4, dim=1)[0])
                X2 = Q3.gather(dim=1, index=action_batch2.long().unsqueeze(dim=1)).squeeze()
                loss2 = loss_fn2(X2, Y2.detach())
                print(i, loss2.item())
                optimizer2.zero_grad()
                loss2.backward()
                losses2.append(loss2.item())
                optimizer2.step()

                if cnt % sync_freq == 0:
                    model4.load_state_dict(model3.state_dict())
            
            print("Reward:",reward)
            total_reward += reward
            accepted_slices_train += accepted_value_train
            total_requests_train += requests_value_train 

        total_reward_list.append(total_reward)
        accepted_train_window.append(accepted_slices_train)
        requests_train_window.append(total_requests_train)
        save_plot_and_csv_train(total_reward_list, accepted_train_window, requests_train_window)

        print("Episode reward:", total_reward)

        if epsilon > 0.01:
            epsilon -= (1 / epochs)

   
    
    #GUARDAR total_reward_list do TRAIN
    torch.save(model1.state_dict(), 'dqn_action1.pt')
    torch.save(model3.state_dict(), 'dqn_action2.pt')
    print("TEST AGENT")
    model_test1=model1
    model_test2=model3

    model_test1.load_state_dict(torch.load("dqn_action1.pt"))
    model_test2.load_state_dict(torch.load("dqn_action2.pt"))
    
    test_rewards=test_model(model_test1,model_test2,env,state_flattened_size)
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
    
     