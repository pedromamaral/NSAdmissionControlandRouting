import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from itertools import count
import torch.nn.functional as F
from matplotlib import pyplot as plt
from numpy import savetxt


"""
    source:https://github.com/gouxiangchen/dueling-DQN-pytorch/blob/master/dueling_dqn.py
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self,l1,l2,l3,l4,l5):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(l1, l2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(l2, l3)
        self.fc_value = nn.Linear(l3, l4)
        self.fc_adv = nn.Linear(l3, l4)

        self.value = nn.Linear(l4, 1)
        self.adv = nn.Linear(l4, l5)

    def forward(self, state):
        y = self.relu(self.fc1(state))
        y = self.relu(self.fc2(y))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()

class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

def test_model(model1,model2, env,l1):
    _state = env.reset()
    state = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, l1)
    done = False
    rewards = []
    while not done:
        # Dueling DQN
        action = model1.select_action(state),model2.select_action(state)
        _state, reward, done, _ = env.step(action)
        state = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, l1)
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
    plt.savefig('dueling_dqn_train.png')
    # save to csv file
    savetxt('dueling_dqn_train.csv', total_rewards, delimiter=',')

def save_plot_and_csv_test(total_rewards):
    x = np.arange(len(total_rewards))
    y = total_rewards
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Reward", fontsize=22)
    plt.plot(x, y, c='black')
    plt.savefig('dueling_dqn_test.png')
    # save to csv file
    savetxt('dueling_dqn_test.csv', total_rewards, delimiter=',')


def dueling_dqn_agent(gamma=0.99, epochs = 4000,final_eps=0.0001, lr=1e-4,eps=0.5, explore=20000, update_step=16, batch_size=256,
         max_memory_size=50000, l1 = 845, l2 = 1500, l3 = 700,l4 = 200, output1 = 2,output2=7,env=""):
    """
    :param gamma: reward discount factor

    :param eps: probability to take a random action during training
    
    :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a
    batch of size "batch_size" from the memory.
    :param batch_size: see above
    :param max_memory_size: size of the replay memory
    :param env_name: name of the gym environment
    :return: 
    """

    env = env
    n_state = l1

#***********************************************************************************
    n_action1 = output1
    
    onlineQNetwork1 = QNetwork(l1,l2,l3,l4,output1).to(device)
    targetQNetwork1 = QNetwork(l1,l2,l3,l4,output1).to(device)
    targetQNetwork1.load_state_dict(onlineQNetwork1.state_dict())
    
    optimizer1 = torch.optim.Adam(onlineQNetwork1.parameters(), lr)
    memory_replay1 = Memory(max_memory_size)
    learn_steps1 = 0
    epsilon1 = eps
    begin_learn1 = False
#***********************************************************************************
#***********************************************************************************
    n_action2 = output2
    
    onlineQNetwork2 = QNetwork(l1,l2,l3,l4,output2).to(device)
    targetQNetwork2 = QNetwork(l1,l2,l3,l4,output2).to(device)
    targetQNetwork2.load_state_dict(onlineQNetwork2.state_dict())
    
    optimizer2 = torch.optim.Adam(onlineQNetwork2.parameters(), lr)
    memory_replay2 = Memory(max_memory_size)
    learn_steps2 = 0
    epsilon2 = eps
    begin_learn2 = False
#***********************************************************************************

    episode_reward = 0
    total_rewards = []

    for epoch in range(epochs):
        cnt=0
        print("Starting training, epoch:", epoch)
        state = env.reset()
        _state = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1, n_state)
        episode_reward = 0
        for time_steps in range(200):
            print("Step:", cnt + 1)
            cnt += 1
            p = random.random()
            if p < epsilon1:
                action1=random.randint(0, output1 - 1)
                action2=random.randint(0, output2 - 1)
                action = action1,action2
            else:
                tensor_state = _state.to(device)
                action1=onlineQNetwork1.select_action(tensor_state)
                action2=onlineQNetwork2.select_action(tensor_state)
                action = action1,action2
            next_state, reward, done, _ = env.step(action)
            print("Reward:",reward)

            _next_state = torch.flatten(torch.from_numpy(next_state.astype(np.float32))).reshape(1, n_state)
            episode_reward += reward


            memory_replay1.add((_state, _next_state, action1, reward, done))
            if memory_replay1.size() > batch_size:
                if begin_learn1 is False:
                    print('learn begin!')
                    begin_learn1 = True
                learn_steps1 += 1
                if learn_steps1 % update_step == 0:
                    targetQNetwork1.load_state_dict(onlineQNetwork1.state_dict())
                batch1 = memory_replay1.sample(batch_size, False)
                batch_state1, batch_next_state1, batch_action1, batch_reward1, batch_done1 = zip(*batch1)

                batch_state1 = torch.cat([item for item in batch_state1]).to(device)
                batch_next_state1 = torch.cat([item for item in batch_next_state1]).to(device)
                batch_action1 = torch.FloatTensor(batch_action1).unsqueeze(1).to(device)
                batch_reward1 = torch.FloatTensor(batch_reward1).unsqueeze(1).to(device)
                batch_done1 = torch.FloatTensor(batch_done1).unsqueeze(1).to(device)

                with torch.no_grad():
                    onlineQ_next1 = onlineQNetwork1(batch_next_state1)
                    targetQ_next1 = targetQNetwork1(batch_next_state1)
                    online_max_action1 = torch.argmax(onlineQ_next1, dim=1, keepdim=True)
                    y1 = batch_reward1 + (1 - batch_done1) * gamma * targetQ_next1.gather(1, online_max_action1.long())

                loss1 = F.mse_loss(onlineQNetwork1(batch_state1).gather(1, batch_action1.long()), y1)
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()

                if epsilon1 > final_eps:
                    epsilon1 -= (eps - final_eps) / explore
            

            memory_replay2.add((_state, _next_state, action2, reward, done))
            if memory_replay2.size() > batch_size:
                if begin_learn2 is False:
                    print('learn begin!')
                    begin_learn2 = True
                learn_steps2 += 1
                if learn_steps2 % update_step == 0:
                    targetQNetwork2.load_state_dict(onlineQNetwork2.state_dict())
                batch2 = memory_replay2.sample(batch_size, False)
                batch_state2, batch_next_state2, batch_action2, batch_reward2, batch_done2 = zip(*batch2)

                batch_state2 = torch.cat([item for item in batch_state2]).to(device)
                batch_next_state2 = torch.cat([item for item in batch_next_state2]).to(device)
                batch_action2 = torch.FloatTensor(batch_action2).unsqueeze(1).to(device)
                batch_reward2 = torch.FloatTensor(batch_reward2).unsqueeze(1).to(device)
                batch_done2 = torch.FloatTensor(batch_done2).unsqueeze(1).to(device)

                with torch.no_grad():
                    onlineQ_next2 = onlineQNetwork2(batch_next_state2)
                    targetQ_next2 = targetQNetwork2(batch_next_state2)
                    online_max_action2 = torch.argmax(onlineQ_next2, dim=1, keepdim=True)
                    y2 = batch_reward2 + (1 - batch_done2) * gamma * targetQ_next2.gather(1, online_max_action2.long())

                loss2 = F.mse_loss(onlineQNetwork2(batch_state2).gather(1, batch_action2.long()), y2)
                optimizer2.zero_grad()
                loss2.backward()
                optimizer2.step()

                if epsilon2 > final_eps:
                    epsilon2 -= (eps - final_eps) / explore

            if done:
                total_rewards.append(episode_reward)
                save_plot_and_csv_train(total_rewards)
                break
            _state = _next_state
        
        print("Episode reward:", episode_reward)
        if epoch % 10 == 0:
            print("TRAIN TOTAL REWARDS",total_rewards)
            torch.save(onlineQNetwork1.state_dict(), 'duelingdqn_action1.pt')
            torch.save(onlineQNetwork2.state_dict(), 'duelingdqn_action2.pt')
            print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))

    
    print("TEST AGENT")
    model_test1=QNetwork(l1,l2,l3,l4,output1).to(device)
    model_test2=QNetwork(l1,l2,l3,l4,output2).to(device)
    model_test1.load_state_dict(torch.load("duelingdqn_action1.pt"))         
    model_test2.load_state_dict(torch.load("duelingdqn_action2.pt"))    
    test_rewards=test_model(model_test1,model_test2,env,l1)      
    save_plot_and_csv_test(test_rewards)
            
"""
    #guardar total_rewards do TRAIN
    print("depois de tudo TRAIN TOTAL REWARDS",total_rewards)

    
    #***************************PLOT TRAIN AND TEST*******************************************
    epochs_array=[]
    rewards_array=[]
    for epochs_n, epochs_reward in enumerate(total_rewards):
        epochs_array.append(epochs_n)
        rewards_array.append(epochs_reward)

    plt.figure(figsize=(10, 7))
    plt.plot(epochs_array,rewards_array,linewidth=2.0)
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Reward", fontsize=22)
    plt.savefig('dueling_dqn_train.png')
    # save to csv file
    savetxt('dueling_dqn_train.csv', total_rewards, delimiter=',')
    
    steps_array=[]
    rewards_array=[]
    for epochs_n, epochs_reward in enumerate(test_rewards):
        steps_array.append(epochs_n)
        rewards_array.append(epochs_reward)

    plt.figure(figsize=(10, 7))
    plt.plot(steps_array,rewards_array,linewidth=2.0)
    plt.xlabel("Steps", fontsize=22)
    plt.ylabel("Reward", fontsize=22)
    plt.savefig('dueling_dqn_test.png')
    # save to csv file
    savetxt('dueling_dqn_test.csv', test_rewards, delimiter=',')

"""