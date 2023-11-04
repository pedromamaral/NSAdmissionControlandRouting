import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import copy
from matplotlib import pyplot as plt
from numpy import savetxt
import csv
#%matplotlib inline

CNT_WINDOW_I = []
CNT_WINDOW_E = []

from Agents.dqn_agent import Agent
from Agents.network import QNetwork

from Env.Network_Path_Selection import ContainernetEnv
from Env.Parameters_NPS import INPUT_DIM,HL1, HL2, HL3, OUTPUT_DIM1,OUTPUT_DIM2,\
    GAMMA, EPSILON, LEARNING_RATE,EPOCHS, MEM_SIZE, BATCH_SIZE, SYNC_FREQ

env = ContainernetEnv()



def save_plot_and_csv_train(total_rewards, accepted_e_window, accepted_i_window, requests_window):

    x = np.arange(len(total_rewards))
    y = total_rewards
    
    #print('\n\n')
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Reward", fontsize=22)
    plt.plot(x, y, c='black')
    plt.savefig('dqn_train.png')
    # save to csv file
    #with open('Train.csv', 'w') as csvfile:
        #fieldnames = ['Rewards', 'Requests', 'Accepted']
        #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        #writer.writeheader()
        #writer.writerow({'Rewards': total_rewards, 'Requests': requests_window, 'Accepted': accepted_window})
    savetxt('dqn_train.csv', total_rewards, delimiter=',')
    savetxt('accepted_e.csv', accepted_e_window, delimiter=',')
    savetxt('accepted_i.csv', accepted_i_window, delimiter=',')
    savetxt('requests.csv', requests_window, delimiter=',')
    with open('Failed_elastic.txt', 'a') as file:
        file.write('\nHello\n')

    with open('Failed_inelastic.txt', 'a') as file2:
        file2.write('\nHello\n')
    #savetxt('Failed_elastic.csv', CNT_WINDOW_E, delimiter=',')
    #savetxt('Failed_inelastic.csv', CNT_WINDOW_I, delimiter=',')


agent = Agent(state_size = 746, action_size = 7, seed = 0)
def train (n_episodes = 4000, max_t = 50, eps_start = 1.0, eps_end = 0.01, eps_decay = 0.999):
    scores = []
    scores_window = deque()
    accepted_eslices: int = 0
    accepted_islices: int = 0
    total_requests: int = 0
    accepted_eslices_epoch: int = 0
    accepted_islices_epoch: int = 0
    total_requests_epoch: int = 0
    accepted_e_window = deque()
    accepted_i_window = deque()
    requests_window = deque()
    #reward_epochs = deque()
    #average_score_epoch = deque()
    #rewards = deque()
    eps = eps_start
    for i_episode in range (1, n_episodes+1):
        print('\n\n\n\n\n\nEPISODE: ', i_episode)
        print('\nEpsilon: ', eps)
        state = env.reset()
        score = 0
        accepted_eslices_epoch = 0
        accepted_islices_epoch = 0
        total_requests_epoch = 0 
        #scores = []
        #average_score_epoch = []
        
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, accepted_eslices, accepted_islices, total_requests, _  = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            accepted_eslices_epoch += accepted_eslices 
            accepted_islices_epoch += accepted_islices
            total_requests_epoch += total_requests 
            
            #rewards.append(reward)
            #average_score_epoch.append(reward)
            
            if done:
                break
        scores_window.append(score)
        accepted_e_window.append(accepted_eslices_epoch)
        accepted_i_window.append(accepted_islices_epoch)
        requests_window.append(total_requests_epoch)
        #scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        #print('\n\rEpisode {}\tAverage Score: {:.2f}\n'.format(i_episode, np.mean(scores_window)), end="")
        #print('Episode reward: ', reward)
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        #print('\nIt reached the end\n')
        #save_plot_and_csv_train2(rewards)
            

        #reward_epochs.append(np.mean(average_score_epoch))
        #average_score_epoch.append(np.mean(scores_window))


        
        save_plot_and_csv_train(scores_window, accepted_e_window, accepted_i_window, requests_window)
        
            

    return scores
scores = train()
