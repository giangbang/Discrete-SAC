import gymnasium as gym

def make_atari(env_id, seed=None):
    env = gym.make(env_id)
    from gym.wrappers.atari_preprocessing import AtariPreprocessing
    env = AtariPreprocessing(env, terminal_on_life_loss=False, scale_obs=True)
    env = gym.wrappers.FrameStack(env, 4)
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env
