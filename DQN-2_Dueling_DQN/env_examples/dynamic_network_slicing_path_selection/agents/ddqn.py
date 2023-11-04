import gym
import torch
import numpy as np
from torch import nn
import random
import torch.nn.functional as F
import collections
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
from numpy import savetxt

"""
Implementation of Double DQN for gym environments with discrete action space.
source: https://github.com/fschur/DDQN-with-PyTorch-for-OpenAI-Gym/blob/master/DDQN_discrete.py
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""
class QNetwork(nn.Module):
    def __init__(self, l1, l2, l3 ,l4, l5):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(l1, l2)
        self.fc_2 = nn.Linear(l2, l3)
        self.fc_3 = nn.Linear(l3, l4)
        self.fc_4 = nn.Linear(l4, l5)

    def forward(self, inp):

        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = F.leaky_relu(self.fc_3(x1))
        x1 = self.fc_4(x1)

        return x1

"""
memory to save the state, action, reward sequence from the current episode. 
"""
class Memory(object):
    def __init__(self, len):
        self.rewards = collections.deque(maxlen=len)
        self.state = collections.deque(maxlen=len)
        self.action = collections.deque(maxlen=len)
        self.is_done = collections.deque(maxlen=len)

    def update(self, state, action, reward, done):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        if not done:
            self.state.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        n = len(self.is_done)
        idx = random.sample(range(0, n-1), batch_size)

        return torch.Tensor(self.state)[idx].to(device), torch.LongTensor(self.action)[idx].to(device), \
               torch.Tensor(self.state)[1+np.array(idx)].to(device), torch.Tensor(self.rewards)[idx].to(device), \
               torch.Tensor(self.is_done)[idx].to(device)

    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()

def select_action_train(model1,model2, state, eps,l1,output1,output2):
    _state = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1, l1)
    state = _state.to(device)

    with torch.no_grad():
        values1 = model1(state)
        values2 = model2(state)

    # select a random action wih probability eps
    if random.random() <= eps:
        action = np.random.randint(0, output1),np.random.randint(0, output2)
    else:
        action = np.argmax(values1.cpu().numpy()),np.argmax(values2.cpu().numpy())

    return action

def select_action_test(model1,model2, state):
    
     with torch.no_grad():
        values1 = model1(state)
        values2 = model2(state)
     action = np.argmax(values1.cpu().numpy()),np.argmax(values2.cpu().numpy())

     return action  

def train(batch_size, current, target, optim, memory, gamma,l1):

    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    _states = states.reshape(batch_size, l1)
    _next_states = next_states.reshape(batch_size, l1)

    q_values = current(_states)

    next_q_values = current(_next_states)
    next_q_state_values = target(_next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()

def evaluate(Qmodel1,Qmodel2, env, repeats,l1):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    
    Qmodel1.eval()
    Qmodel2.eval()
    perform = 0
    for i in range(repeats):
        print("Starting evaluate, epoch:", i)
        cnt=0
        state = env.reset()
        done = False
        while not done:
            print("Step evaluate:", cnt + 1)
            cnt += 1
            _state = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1, l1).to(device)
            with torch.no_grad():
                values1 = Qmodel1(_state)
                values2 = Qmodel2(_state)
                print("Evaluate CHOICES",values1)
                print("Evaluate PATHS",values2)
            action = np.argmax(values1.cpu().numpy()),np.argmax(values2.cpu().numpy())
            state, reward, done, _ = env.step(action)
            print("Reward evaluate:",reward)
            perform += reward
    Qmodel1.train()
    Qmodel2.train()
    return perform/repeats

def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def test_model(model1,model2, env,l1):
    _state = env.reset()
    state = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, l1)
    done = False
    rewards = []
    while not done:
        # DDQN
        action = select_action_test(model1,model2,state)
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
    plt.savefig('ddqn_train.png')
    # save to csv file
    savetxt('ddqn_train.csv', total_rewards, delimiter=',')

def save_plot_and_csv_test(total_rewards):
    x = np.arange(len(total_rewards))
    y = total_rewards
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Reward", fontsize=22)
    plt.plot(x, y, c='black')
    plt.savefig('ddqn_test.png')
    # save to csv file
    savetxt('ddqn_test.csv', total_rewards, delimiter=',')



def ddqn_agent(gamma=0.99, lr=1e-3, min_episodes=20, eps=1, eps_decay=0.995, eps_min=0.01, update_step=10, batch_size=64, update_repeats=50,
         epochs=4000, seed=42, max_memory_size=50000, lr_gamma=0.9, lr_step=100, measure_step=100,
         measure_repeats=100,l1 = 845, l2 = 1500, l3 = 700,l4 = 200, output1=2,output2=7,  env=""):

    """
    :param gamma: reward discount factor
    :param lr: learning rate for the Q-Network
    :param min_episodes: we wait "min_episodes" many episodes in order to aggregate enough data before starting to train
    :param eps: probability to take a random action during training
    :param eps_decay: after every episode "eps" is multiplied by "eps_decay" to reduces exploration over time
    :param eps_min: minimal value of "eps"
    :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a
    batch of size "batch_size" from the memory.
    :param batch_size: see above
    :param update_repeats: see above
    :param num_episodes: the number of episodes played in total
    :param seed: random seed for reproducibility
    :param max_memory_size: size of the replay memory
    :param lr_gamma: learning rate decay for the Q-Network
    :param lr_step: every "lr_step" episodes we decay the learning rate
    :param measure_step: every "measure_step" episode the performance is measured
    :param measure_repeats: the amount of episodes played in to asses performance
    :param hidden_dim: hidden dimensions for the Q_network
    :param env_name: name of the gym environment
    :return: 
    """
    
    total_rewards = []
    env = env
    torch.manual_seed(seed)
    env.seed(seed)
    performance = []
    
#***********************************************************************************
    Q_1 = QNetwork(l1,l2,l3,l4,output1).to(device)
    Q_2 = QNetwork(l1,l2,l3,l4,output1).to(device)
        
    # transfer parameters from Q_1 to Q_2
    update_parameters(Q_1, Q_2)

    # we only train Q_1
    for param in Q_2.parameters():
        param.requires_grad = False

    optimizer1 = torch.optim.Adam(Q_1.parameters(), lr=lr)
    optimizer1.step()
    
    scheduler1 = StepLR(optimizer1, step_size=lr_step, gamma=lr_gamma)

    memory1 = Memory(max_memory_size)
#***********************************************************************************
#***********************************************************************************
    Q_3 = QNetwork(l1,l2,l3,l4,output2).to(device)
    Q_4 = QNetwork(l1,l2,l3,l4,output2).to(device)
        
    # transfer parameters from Q_1 to Q_2
    update_parameters(Q_3, Q_4)

    # we only train Q_1
    for param in Q_4.parameters():
        param.requires_grad = False

    optimizer2 = torch.optim.Adam(Q_3.parameters(), lr=lr)
    optimizer2.step()
    
    scheduler2 = StepLR(optimizer2, step_size=lr_step, gamma=lr_gamma)

    memory2 = Memory(max_memory_size)
#***********************************************************************************
    


    for episode in range(epochs):
        print("Starting training, epoch:", episode)
        
        # display the performance
        if episode % measure_step == 0:
            performance.append([episode, evaluate(Q_1,Q_3,env, measure_repeats,l1)])
            print("Episode: ", episode)
            print("rewards: ", performance[-1][1])
            total_rewards.append(performance[-1][1])
            save_plot_and_csv_train(total_rewards)
            print("lr1: ", scheduler1.get_last_lr()[0])
            print("lr2: ", scheduler2.get_last_lr()[0])
            print("eps: ", eps)

        state = env.reset()
        memory1.state.append(state.reshape(1, l1))
        memory2.state.append(state.reshape(1, l1))

        done = False
        i = 0
        while not done:
            print("Step:", i + 1)
            i += 1
            action=select_action_train(Q_2,Q_4, state, eps,l1,output1,output2)
            action1 = action[0]
            action2 = action[1]
            state, reward, done, _ = env.step(action)
            print("Reward:",reward)
            # save state, action, reward sequence
            memory1.update(state.reshape(1, l1), action1, reward, done)
            memory2.update(state.reshape(1, l1), action2, reward, done)

        if episode >= min_episodes and episode % update_step == 0:
            for _ in range(update_repeats):
                train(batch_size, Q_1, Q_2, optimizer1, memory1, gamma,l1)
                train(batch_size, Q_3, Q_4, optimizer2, memory2, gamma,l1)

            # transfer new parameter from Q_1 to Q_2
            update_parameters(Q_1, Q_2)
            update_parameters(Q_3, Q_4)

        # update learning rate and eps
        scheduler1.step()
        scheduler2.step()
        eps = max(eps*eps_decay, eps_min)


    #Q_1, performance, total_rewards
    #guardar total_rewards do TRAIN
    print("TRAIN TOTAL REWARDS",total_rewards)
    torch.save(Q_1.state_dict(), 'ddqn_action1.pt')
    torch.save(Q_3.state_dict(), 'ddqn_action2.pt')

    print("TEST AGENT")
    
    model_test1=QNetwork(l1,l2,l3,l4,output1).to(device)
    model_test2=QNetwork(l1,l2,l3,l4,output2).to(device)

    model_test1.load_state_dict(torch.load("ddqn_action1.pt"))
    model_test2.load_state_dict(torch.load("ddqn_action2.pt"))
    
    test_rewards=test_model(model_test1,model_test2,env,l1) 
    save_plot_and_csv_test(test_rewards)
    """
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
    plt.savefig('ddqn_train.png')
    # save to csv file
    savetxt('ddqn_train.csv', total_rewards, delimiter=',')
    
    steps_array=[]
    rewards_array=[]
    for epochs_n, epochs_reward in enumerate(test_rewards):
        steps_array.append(epochs_n)
        rewards_array.append(epochs_reward)

    plt.figure(figsize=(10, 7))
    plt.plot(steps_array,rewards_array,linewidth=2.0)
    plt.xlabel("Steps", fontsize=22)
    plt.ylabel("Reward", fontsize=22)
    plt.savefig('ddqn_test.png')
    # save to csv file
    savetxt('ddqn_test.csv', test_rewards, delimiter=',')
    
"""
    
