# Direct Behavior Specification via Constrained RL 

Code to reproduce the Arena environment experiments from [Direct Behavior Specification via Constrained Reinforcement Learning](https://arxiv.org/abs/2112.12228). See installation and run procedures below.

## License

Please read the [license](./LICENSE.txt).
Here's a [summary](https://creativecommons.org/licenses/by-nc-nd/4.0/).


## Installation

* Create a conda environment: `conda create --name dbs python=3.8.8`
* Install dependencies: `pip install -r requirements.txt`

## To train a model

Simply run `main.py` with the desired arguments.

examples:

##### SAC with Reward Engineering
```
python main.py --constraints_to_enforce is-looking-at-marker is-in-lava is-above-energy-limit --constraint_is_reversed true false true --constraint_fixed_weights 0.25 2. 0.5 --constraint_discount_factors 0.9 0.9 0.9 --constraint_rates_to_add_as_obs is-looking-at-marker is-in-lava is-above-energy-limit --constraint_enforcement_method reward_engineering --steps_bw_update 200 --num_steps 5000000 --desc rewardEngineering
```

##### SAC-Lagrangian with single constraints
```
python main.py --constraints_to_enforce is-above-energy-limit --constraint_is_reversed true --constraint_enforcement_method lagrangian --constraint_thresholds nan-0.01 --constraint_discount_factors 0.9 --constraint_rates_to_add_as_obs is-above-energy-limit --num_steps 3000000 --desc singleConstraintEnergy
```

```
python main.py --constraints_to_enforce is-on-ground --constraint_is_reversed true --constraint_enforcement_method lagrangian --constraint_thresholds nan-0.40 --constraint_discount_factors 0.9 --constraint_rates_to_add_as_obs is-on-ground --num_steps 3000000 --desc singleConstraintJump
```

```
python main.py --constraints_to_enforce is-in-lava --constraint_is_reversed false --constraint_enforcement_method lagrangian --constraint_thresholds nan-0.01 --constraint_discount_factors 0.9 --constraint_rates_to_add_as_obs is-in-lava --num_steps 3000000 --desc singleConstraintLava
```

```
python main.py --constraints_to_enforce is-looking-at-marker --constraint_is_reversed true --constraint_enforcement_method lagrangian --constraint_thresholds nan-0.10 --constraint_discount_factors 0.9 --constraint_rates_to_add_as_obs is-looking-at-marker --num_steps 3000000 --desc singleConstraintLookat
```

```
python main.py --constraints_to_enforce is-above-speed-limit --constraint_is_reversed false --constraint_enforcement_method lagrangian --constraint_thresholds nan-0.01 --constraint_discount_factors 0.9 --constraint_rates_to_add_as_obs is-above-speed-limit --num_steps 3000000 --desc singleConstraintSpeed
```

##### SAC-Lagrangian with multiple constraints
```
python main.py --constraints_to_enforce has-reached-goal-in-episode is-looking-at-marker is-on-ground is-in-lava is-above-speed-limit is-above-energy-limit --constraint_is_reversed false true true false false true --constraint_thresholds 0.99-nan,nan-0.1,nan-0.4,nan-0.01,nan-0.01,nan-0.01 --constraint_discount_factors 0.9 0.9 0.9 0.9 0.9 0.9 --constraint_rates_to_add_as_obs is-looking-at-marker is-on-ground is-in-lava is-above-speed-limit is-above-energy-limit --bootstrap_constraint has-reached-goal-in-episode --constraint_enforcement_method lagrangian --num_steps 10000000 --desc allConstraints
```

## To visualise a model

Simply run `evaluate.py` the appropriate arguments.

example:
```
python evaluate.py --root_dir storage --storage_name No4_sac_ArenaEnv-v0_singleConstraintLava --max_episode_len 100 --n_episodes 10 --render true
```

## Bibtex

```
@article{roy2021direct,
  title={Direct Behavior Specification via Constrained Reinforcement Learning},
  author={Roy, Julien and Girgis, Roger and Romoff, Joshua and Bacon, Pierre-Luc and Pal, Christopher},
  journal={arXiv preprint arXiv:2112.12228},
  year={2021}
}
```

© [2022] Ubisoft Entertainment. All Rights Reserved
