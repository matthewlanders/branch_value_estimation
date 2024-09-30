This repository contains the official implementation of the forthcoming paper, Offline Reinforcement Learning with Combinatorial Action Spaces.
## How to run the code

### Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

### Generate offline data
```
python agents/a_star_agent.py --grid_dimension 10 --grid_size 5 --num_pits 5 --num_clusters 1 --save_transitions
```
This data generation command will print a list of terminal states (goal state and pits) to the terminal. Use this output to update the `terminal_states_dict` in `common.py`.

### Run training/evaluation
```
python agents/bve_agent.py --data_load_path offline_data/10-5-5-1-0.9 --q_loss_multiplier 20 --num_beams 200 --num_network_layers 2 --lr_decay_rate 0.99995
```
