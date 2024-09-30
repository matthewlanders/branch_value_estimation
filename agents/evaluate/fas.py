import numpy as np
import torch


def evaluate(env, eval_episodes, agent):
    obs = env.reset()
    episodic_returns = []
    episode_actions = []

    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            obs = torch.tensor(obs).unsqueeze(0)
            action = agent.get_action(obs.float(), batch_size=1)
            action = np.ravel_multi_index(tuple(action.cpu().squeeze().numpy().tolist()), (2,) * 2 * env.grid_dimension)
            episode_actions.append(action)

        next_obs, rewards, terminations, info = env.step(action)

        if "final_info" in info:
            data = info['final_info']['episode']
            episodic_returns += [data['r']]

        obs = next_obs

    return np.array(episodic_returns)
