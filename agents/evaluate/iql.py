import numpy as np
import torch


def evaluate(env, eval_episodes, agent):
    obs = env.reset()
    episodic_returns = []

    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            action = agent.get_action(obs)

        next_obs, rewards, terminations, info = env.step(action)

        if "final_info" in info:
            data = info['final_info']['episode']
            episodic_returns += [data['r']]

        obs = next_obs

    return np.array(episodic_returns)
