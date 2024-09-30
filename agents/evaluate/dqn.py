import numpy as np
import torch


def evaluate(env, eval_episodes, network, device, actions_map):
    obs = env.reset()
    episodic_returns = []
    episode_actions = []

    while len(episodic_returns) < eval_episodes:
        q_values = network(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=0).cpu().numpy()
        actions = actions_map[actions]
        episode_actions.append(actions)

        next_obs, rewards, terminations, info = env.step(actions)

        if "final_info" in info:
            data = info['final_info']['episode']
            episodic_returns += [data['r']]

        obs = next_obs

    return np.array(episodic_returns)
