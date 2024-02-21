from gymnasium.envs.registration import register

register(
    id='uav-v1',
    entry_point='gym_uav.envs:UavDenseEnv',
    max_episode_steps=100
)
