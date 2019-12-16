# Distributional Learning Applied to Safe Exploration

This repository contains the code for the Columbia Fall 2019 course COMS 6998 Topics in Robotic Learning. The goal of the project is to develop safe RL approaches to deal with safe exploration in robotic robotic learning. The code under `ai_safety_gridworlds` is forked from [this](https://github.com/deepmind/ai-safety-gridworlds) repository.

## How To Run
Install the following dependencies:
* [MuJoCo](https://github.com/openai/mujoco-py)
* [Safety Gym](https://github.com/openai/safety-gym)
* `pip install -r requirements.txt`

Then you should be able to run the CPPO (Constrained PPO) agent like so

`python3 -m agents.cppo <options>`

Arguments for the script are saved in `args.json` files under the `results/` tree. Copying the arguments from these should enable you to reproduce the results.

For example, to reproduce the run from `results/pointgoal1/zero_001` we run

```bash
python3 -m agents.cppo \
  --env_name Safexp-PointGoal1-v0 \
  --risk_type zero \
  --steps_per_iter 8000 \
  --batch_size 8000 \
  --num_train_iters 200 \
  --save_interval 1 \
  --seed 0 \
  --run_dir runs/zero_001 \
  --target_kl 0.02 \
  --stop_extra_early
```
