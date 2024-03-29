'''This file is not well-maintained and contains ugly code
since it should not be copied to other projects
'''

from sac_discrete import ReplayBuffer, SACDiscrete as SAC
import numpy as np
import gymnasium as gym
from utils import parse_args, pprint, seed_everything
from logger import Logger

def evaluate(env, agent, n_rollout = 10):
    tot_rw = 0
    for _ in range(n_rollout):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, True)

            next_state, reward, done, info = env.step(action)
            tot_rw += reward
            state = next_state
            done = terminated or truncated
    return tot_rw / n_rollout


def main():
    args = parse_args()
    logger = Logger()
    logger.add_run_command()

    if args.seed > 0: seed_everything(args.seed)

    env = gym.make(args.env_name)

    action_shape      = env.action_space.n
    observation_shape = env.observation_space.shape

    sac_agent = SAC(observation_shape[0], action_shape, **vars(args))
    buffer    = ReplayBuffer(observation_shape, [action_shape],
                args.buffer_size, args.batch_size)

    pprint(vars(args))
    print('Action dim: {} | Observation dim: {}'.format(action_shape, observation_shape))

    his = []
    loss = []

    train_returns = 0

    state, _ = env.reset()
    for env_step in  range(int(args.total_env_step)):
        if env_step < args.start_step:
            action = env.action_space.sample()
        else :
            action = sac_agent.select_action(state, False)

        next_state, reward, terminated, truncated, info = env.step(action)
        buffer.add(state, action, reward, next_state, done, info)
        train_returns += reward

        if (env_step + 1) % args.train_freq == 0:
            loss.append(sac_agent.update(buffer))

        state = next_state
        done = terminated or truncated
        if done:
            state, _ = env.reset()
            logger.add_scalar("train/returns", train_returns, env_step)
            train_returns = 0
        if (env_step + 1) % args.eval_interval == 0:
            eval_return = evaluate(gym.make(args.env_name), sac_agent, args.num_eval_episodes)
            his.append(eval_return)
            print('mean reward after {} env step: {:.2f}'.format(env_step+1, eval_return))
            print('critic loss: {:.2f} | actor loss: {:.2f} | alpha loss: {:.2f} | alpha: {:.2f}'.format(
                    *np.mean(list(zip(*loss[-10:])), axis=-1),
                    sac_agent.log_ent_coef.exp().item()
                    ))
            logger.add_scalar("eval/returns", eval_return, env_step, smooth=False)

    logger.close()

    import matplotlib.pyplot as plt
    x, y = np.linspace(0, args.total_env_step, len(his)), his
    plt.plot(x, y)
    plt.title(args.env_name)
    plt.savefig('res.png')

    import pandas as pd
    data_dict = {'rollout/ep_rew_mean': y, 'time/total_timesteps': x} # formated as stable baselines
    df = pd.DataFrame(data_dict)

    df.to_csv('sac_discrete_progress.csv', index=False)
    sac_agent.save('model', args.total_env_step)

if __name__ == '__main__':
    main()
