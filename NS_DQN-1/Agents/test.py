import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
#%matplotlib inline
from numpy import savetxt


from Agents.dqn_agent import Agent
from Agents.network import QNetwork

from Env.Network_Path_Selection import ContainernetEnv
from Env.Parameters_NPS import INPUT_DIM,HL1, HL2, HL3, OUTPUT_DIM1,OUTPUT_DIM2,\
    GAMMA, EPSILON, LEARNING_RATE,EPOCHS, MEM_SIZE, BATCH_SIZE, SYNC_FREQ

env = ContainernetEnv()
agent = Agent(state_size = 746, action_size = 7, seed = 0)



def save_plot_and_csv_test(total_rewards, accepted_window, requests_window):

    x = np.arange(len(total_rewards))
    y = total_rewards
    
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Reward", fontsize=22)
    plt.plot(x, y, c='black')
    plt.savefig('dqn_test.png')
    
    savetxt('dqn_test.csv', total_rewards, delimiter=',')
    savetxt('accepted_test.csv', accepted_window, delimiter=',')
    savetxt('requests_test.csv', requests_window, delimiter=',')

    with open('Failed_elastic.txt', 'a') as file:
        file.write('\n\n\n')

    with open('Failed_inelastic.txt', 'a') as file2:
        file2.write('\n\n\n')

    

agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
def test():
    scores_window = deque()
    accepted_slices_window = deque()
    total_requests_window = deque()
    for i in range(100):
        score = 0
        accepted = 0
        requests = 0
        state = env.reset()
        for j in range (50):
            action = agent.act(state)
            next_state, reward, done, accepted_slices, total_requests, _  = env.step(action)
            score = score + reward
            accepted = accepted + accepted_slices
            requests = requests + total_requests
            if done:
                break
        scores_window.append(score)
        accepted_slices_window.append(accepted)
        total_requests_window.append(requests)
        save_plot_and_csv_test(scores_window, accepted_slices_window, total_requests_window)

