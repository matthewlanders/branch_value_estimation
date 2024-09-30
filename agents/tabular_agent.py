import random
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from grid_world import GridWorld


def linear_schedule(start, end, rate, step):
    return max(end, start - step * (start - end) / rate)


def get_best_action(env, q_table, obs):
    q_values = q_table[tuple(obs)]
    if q_values:
        max_value = max(q_values.values())
        max_actions = [action for action, value in q_values.items() if value == max_value]
        return random.choice(max_actions)

    return np.random.randint(env.total_actions)


def learn(env, start_e, end_e, exploration_rate, total_timesteps, gamma, alpha):
    q_table = defaultdict(lambda: defaultdict(float))
    obs = env.reset()
    episode_rewards = []

    for global_step in range(total_timesteps):
        ep = linear_schedule(start_e, end_e, exploration_rate, global_step)
        action = np.random.randint(env.total_actions) if np.random.rand() < ep else get_best_action(env, q_table, obs)

        next_obs, reward, done, info = env.step(action)

        if "final_info" in info:
            returns = info['final_info']['episode']['r']
            print(f"global_step={global_step}, episodic_return={returns}")
            episode_rewards.append(returns)

        old_q_value = q_table[tuple(obs)][action]
        next_max_q = max(q_table[tuple(next_obs)].values(), default=0)
        td_target = reward + gamma * next_max_q * (1 - int(done))
        td_error = td_target - old_q_value

        new_q_value = old_q_value + alpha * td_error
        q_table[tuple(obs)][action] = new_q_value

        obs = next_obs

    return episode_rewards, q_table


if __name__ == "__main__":
    terminal_states = [(4, 4), (1, 1), (2, 1), (3, 3)]
    environment = GridWorld(grid_dimension=2, grid_size=5, num_total_pits=3, num_clusters=1,
                            distribute_pits_evenly=True, only_independent=False, max_steps_per_episode=1000,
                            terminal_states=terminal_states, randomize_initial_state=True)

    reward_per_episode, qt = learn(env=environment, start_e=1, end_e=0.05, exploration_rate=0.5,
                                   total_timesteps=50000, gamma=0.9, alpha=0.1)
    plt.plot(reward_per_episode)
    plt.show()
