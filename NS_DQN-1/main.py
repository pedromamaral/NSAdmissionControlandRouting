import os
#define the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

#function to get paths relative to the working directory
def get_path(relative_path):
    return os.path.join(working_dir,relative_path)


#dynamic_network_slicing_path_selection - 
from Env.Network_Path_Selection import ContainernetEnv
from Env.Parameters_NPS import INPUT_DIM,HL1, HL2, HL3, OUTPUT_DIM1,OUTPUT_DIM2,\
    GAMMA, EPSILON, LEARNING_RATE,EPOCHS, MEM_SIZE, BATCH_SIZE, SYNC_FREQ

from Agents import test
import Agents.test 
import Agents.train

if __name__ == "__main__":
    #env= gym.make("containernet_gym/ContainernetEnv-v0")
    env=ContainernetEnv()
    Agents.train.train()
    #NPS_DNS-DONE
    """ dqn_agent(gamma = GAMMA, epsilon = EPSILON, learning_rate = LEARNING_RATE,state_flattened_size = INPUT_DIM, epochs = EPOCHS,\
        mem_size = MEM_SIZE,batch_size = BATCH_SIZE,sync_freq = SYNC_FREQ,l1 = INPUT_DIM, l2 = HL1, l3 = HL2,l4 = HL3, \
     output1 = OUTPUT_DIM1, output2=OUTPUT_DIM2, env=env) """
   