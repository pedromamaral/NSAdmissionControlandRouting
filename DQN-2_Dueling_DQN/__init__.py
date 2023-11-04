from gym.envs.registration import register

register(
    id='containernet_gym/ContainernetEnv-v0',
    entry_point='containernet_gym.envs:ContainernetEnv'
)