import os
#define the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

#function to get paths relative to the working directory
def get_path(relative_path):
    return os.path.join(working_dir,relative_path)

from agents.ddqn import ddqn_agent
from agents.dqn import dqn_agent
from agents.duelingdqn import dueling_dqn_agent

#---------------------------------------------------------EXAMPLE 1---------------------------------------------------------
#network_path_selection - EXAMPLE 1
#from env_examples.network_path_selection.containernetenv_network_path_selection import ContainernetEnv
#from env_examples.network_path_selection.parameters import INPUT_DIM,HL1, HL2, HL3, OUTPUT_DIM, GAMMA, EPSILON,\

#  LEARNING_RATE,EPOCHS, MEM_SIZE, BATCH_SIZE, SYNC_FREQ
#---------------------------------------------------------EXAMPLE 1---------------------------------------------------------

#---------------------------------------------------------EXAMPLE 2---------------------------------------------------------
#dynamic_network_slicing - EXAMPLE 2
#from env_examples.dynamic_network_slicing.containernetenv_dynamic_network_slicing import ContainernetEnv
#from env_examples.dynamic_network_slicing.parameters import INPUT_DIM,HL1, HL2, HL3, OUTPUT_DIM, GAMMA, EPSILON,\
#     LEARNING_RATE,EPOCHS, MEM_SIZE, BATCH_SIZE, SYNC_FREQ
#---------------------------------------------------------EXAMPLE 2---------------------------------------------------------

#---------------------------------------------------------EXAMPLE 3---------------------------------------------------------
#dynamic_network_slicing_path_selection - 
from env_examples.dynamic_network_slicing_path_selection.containernetenv_dynamic_network_slicing_path_selection import ContainernetEnv
from env_examples.dynamic_network_slicing_path_selection.parameters import INPUT_DIM,HL1, HL2, HL3, OUTPUT_DIM1,OUTPUT_DIM2,\
    GAMMA, EPSILON, LEARNING_RATE,EPOCHS, MEM_SIZE, BATCH_SIZE, SYNC_FREQ

from env_examples.dynamic_network_slicing_path_selection.agents.ddqn import ddqn_agent
from env_examples.dynamic_network_slicing_path_selection.agents.dqn import dqn_agent
from env_examples.dynamic_network_slicing_path_selection.agents.duelingdqn import dueling_dqn_agent
#---------------------------------------------------------EXAMPLE 3---------------------------------------------------------

if __name__ == "__main__":
    #env= gym.make("containernet_gym/ContainernetEnv-v0")
    env=ContainernetEnv()

    #---------------------------------------------------------EXAMPLE 1-2---------------------------------------------------------
    #NPS-DONE
    #DNS-DONE
    #dqn_agent(gamma = GAMMA, epsilon = EPSILON, learning_rate = LEARNING_RATE,state_flattened_size = INPUT_DIM, epochs = EPOCHS,\
    #    mem_size = MEM_SIZE,batch_size = BATCH_SIZE,sync_freq = SYNC_FREQ,l1 = INPUT_DIM, l2 = HL1, l3 = HL2,l4 = HL3, \
    # l5 = OUTPUT_DIM, env=env)

    #NPS-DONE
    #DNS-DONE
    #dueling_dqn_agent(gamma=GAMMA, epochs = EPOCHS,final_eps=0.0001, lr=LEARNING_RATE,eps=EPSILON, explore=20000, update_step=16,\
    #  batch_size=BATCH_SIZE, max_memory_size=MEM_SIZE, l1 = INPUT_DIM, l2 = HL1, l3 = HL2,l4 = HL3, l5 = OUTPUT_DIM,env=env)
    
    #NPS-DONE
    #DNS-DONE
    #ddqn_agent(gamma=GAMMA, lr=LEARNING_RATE, min_episodes=20, eps=EPSILON, eps_decay=0.995, eps_min=0.01, update_step=10,\
    #  batch_size=BATCH_SIZE, update_repeats=50, epochs=EPOCHS, seed=42, max_memory_size=MEM_SIZE, lr_gamma=0.9, lr_step=100,\
    #  measure_step=100, measure_repeats=100,l1 = INPUT_DIM, l2 = HL1, l3 = HL2,l4 = HL3, l5 = OUTPUT_DIM,  env=env)

    #---------------------------------------------------------EXAMPLE 1-2---------------------------------------------------------

    #---------------------------------------------------------EXAMPLE 3---------------------------------------------------------
    #NPS_DNS-DONE
    dqn_agent(gamma = GAMMA, epsilon = EPSILON, learning_rate = LEARNING_RATE,state_flattened_size = INPUT_DIM, epochs = EPOCHS,\
        mem_size = MEM_SIZE,batch_size = BATCH_SIZE,sync_freq = SYNC_FREQ,l1 = INPUT_DIM, l2 = HL1, l3 = HL2,l4 = HL3, \
     output1 = OUTPUT_DIM1, output2=OUTPUT_DIM2, env=env)

    #NPS_DNS-DONE
    #dueling_dqn_agent(gamma=GAMMA, epochs = EPOCHS,final_eps=0.0001, lr=LEARNING_RATE,eps=EPSILON, explore=20000, update_step=16,\
    #  batch_size=BATCH_SIZE, max_memory_size=MEM_SIZE, l1 = INPUT_DIM, l2 = HL1, l3 = HL2,l4 = HL3, output1 = OUTPUT_DIM1, \
    #    output2=OUTPUT_DIM2, env=env)
    
    #NPS_DNS-DONE
    #ddqn_agent(gamma=GAMMA, lr=LEARNING_RATE, min_episodes=20, eps=EPSILON, eps_decay=0.995, eps_min=0.01, update_step=10,\
    #  batch_size=BATCH_SIZE, update_repeats=50, epochs=EPOCHS, seed=42, max_memory_size=MEM_SIZE, lr_gamma=0.9, lr_step=100,\
    #  measure_step=100, measure_repeats=100,l1 = INPUT_DIM, l2 = HL1, l3 = HL2,l4 = HL3, output1 = OUTPUT_DIM1, output2=OUTPUT_DIM2,\
    #  env=env)
    #---------------------------------------------------------EXAMPLE 3---------------------------------------------------------