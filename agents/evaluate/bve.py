import numpy as np
import torch


def compute_children(env, idx):
    action_shape = (2,) * 2 * env.grid_dimension
    current_action = env.compute_action_from_index(idx)
    last_activated_sub_action = np.max(np.where(current_action == 1)[0], initial=-1)

    children = []
    for i in range(last_activated_sub_action + 1, len(current_action)):
        child_action = np.copy(current_action)
        child_action[i] = 1
        child_index = np.ravel_multi_index(tuple(child_action), action_shape)
        children.append(child_index)

    return children


def compute_action(env, network, obs, device, sa, sab, k):
    obs = torch.Tensor(obs).to(device)
    beams = [0]
    beam_values = [-float('inf')]
    beams_to_explore = [0]
    explored_beams = set()

    while beams_to_explore:
        action = beams_to_explore.pop(0)
        if action in explored_beams:
            continue
        explored_beams.add(action)

        children = compute_children(env, action)
        children = [c for c in children if c in sab]

        with torch.no_grad():
            action_tensor = torch.tensor(env.compute_action_from_index(action), device=device).view(1, -1)
            values = network(obs.unsqueeze(0), action_tensor).flatten()[:len(children) + 1]

        # Use Q value instead of BVE
        action_idx = beams.index(action)
        beam_values[action_idx] = values[0].item()

        top_action_values, top_action_indices = torch.topk(values, min(k, len(children) + 1))

        if 0 in top_action_indices:
            if action not in sa:
                masked_values = values.clone()
                masked_values[0] = float('-inf')
                top_action_values, top_action_indices = torch.topk(masked_values, min(k, len(children) + 1))

        for i, action_value in enumerate(top_action_values):
            new_action = children[top_action_indices[i] - 1] if top_action_indices[i] > 0 else action
            if new_action not in explored_beams and new_action not in beams_to_explore:
                if len(beams) == k:
                    if action_value.item() >= min(beam_values):
                        min_action_value_idx = beam_values.index(min(beam_values))
                        action_to_remove = beams[min_action_value_idx]

                        if action_to_remove not in explored_beams:
                            beams_to_explore_idx = beams_to_explore.index(action_to_remove)
                            beams_to_explore.pop(beams_to_explore_idx)

                        beams[min_action_value_idx] = new_action
                        beam_values[min_action_value_idx] = action_value.item()
                        beams_to_explore.append(new_action)

                else:
                    beams.append(new_action)
                    beam_values.append(action_value.item())
                    beams_to_explore.append(new_action)

    max_index = beam_values.index(max(beam_values))
    return np.array([beams[max_index]])


def evaluate(env, eval_episodes, network, device, seen_actions, seen_action_branches, num_beams):
    obs = env.reset()
    episodic_returns = []
    episode_actions = []

    while len(episodic_returns) < eval_episodes:
        actions = compute_action(env, network, obs, device, seen_actions, seen_action_branches, num_beams)
        episode_actions.append(actions)

        next_obs, rewards, terminations, info = env.step(actions)

        if "final_info" in info:
            data = info['final_info']['episode']
            episodic_returns += [data['r']]

        obs = next_obs

    return np.array(episodic_returns)
