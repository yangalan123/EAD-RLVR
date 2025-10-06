# Recipe: Explorative Annealed Decoding for Verifiable Reinforcement Learning (EAD)

Chenghao Yang, Lin Gui, Chenxiao Yang, Victor Veitch, Lizhu Zhang, Zhuokai Zhao 

> Currently the code can only run under **vLLM V0 API**. We do have experiment support of V1 API but has not fully tested yet. 

## Quickstart
1. Follow [Minimal-RL](https://github.com/RLHFlow/Minimal-RL/tree/main?tab=readme-ov-file#experiments-running) to prepare the dataset. 
2. Run `run_annealed_sampling.sh`. 

## Customize Annealed Strategy and Parameter Explanation
Take a read at `verl/workers/rollout/annealed_sampling.py`