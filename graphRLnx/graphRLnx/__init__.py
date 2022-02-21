from gym.envs.registration import register

register(
    id='graphRL-v0',
    entry_point='graphRLnx.envs:graphRLnx',
)