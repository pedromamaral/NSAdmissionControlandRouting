import gym
from mininet.net import Containernet
from gym import Env, spaces


class ContainernetEnv(Env):
    def __init__(self):
        super(ContainernetEnv,self).__init__()
        #define Observation Space
        self.observation_shape
        self.observation_space
        #define Action Space
        self.action_space
        #define Env Elements/Variables
    ''' 
        get_state function
        translates the environment state into an observation
        The agent gets to a new state or observation state is
        the information of the environment that an agent is in
        and observation is an actual image that the agent sees.
    '''
    def get_state(self):
        #print("_get_state function")
    ''' 
        get information function
        return auxiliary information that is returned by step and reset function
    '''

    def _get_info(self):
        print("_get_info function")

    ''' 
        reset function - This function resets the environment to its initial state,
        and returns the observation of the environment corresponding to the initial state.
    '''
    def reset(self):
        print("reset function")        

    ''' 
        step function -This function takes an action as an input and applies it to
        the environment, which leads to the environment transitioning to a new state.
        returns:
            - observation: The observation of the state of the environment.
            - reward: The reward that you can get from the environment after executing 
            the action that was given as the input to the step function.
            - done: Whether the episode has been terminated. If true, you may need to end the 
            simulation or reset the environment to restart the episode.
            - info: This provides additional information depending on the environment,
            such as number of lives left, or general information that may be conducive in debugging.
    '''
    def step(self, action):
        print("step function")

