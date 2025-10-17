
## 🚀 Exploratory Annealed Decoding (EAD)

We're excited to present our latest research contribution: **Exploratory Annealed Decoding (EAD) for Verifiable Reinforcement Learning**!

- 📄 [Research Paper](https://arxiv.org/abs/2510.05251)
- 🌐 [Project Website](https://yangalan123.github.io/ead_rlvr/)

Our codebase is based on [verl](https://github.com/volcengine/verl) RL framework and [vLLM](https://github.com/vllm-project/vllm) inference engine. We made our best efforts to save the commit history so everyone can easily find what we have updated. We are working to integrate our methods to both upstreams. Stay tuned! 

## Citation

If you use this code as part of any published research, please acknowledge the following paper:

```bibtex
@article{yang2025ead,
  title={Let it Calm: Exploratory Annealed Decoding for Verifiable Reinforcement Learning},
  author={Yang, Chenghao and Gui, Lin and Yang, Chenxiao and Veitch, Victor and Zhang, Lizhu and Zhao, Zhuokai},
  journal={arXiv preprint arXiv:2510.05251},
  year={2025}
}
```

## What is EAD?

EAD is a simple yet effective exploration strategy for Reinforcement Learning with Verifiable Rewards (RLVR) that addresses a fundamental challenge: achieving effective exploration while preserving sample quality and ensuring training stability.

**Core Insight**: Exploration is not equally valuable at every step. Early tokens shape a sequence's semantic direction, making early exploration crucial for discovering diverse valid solutions. Later tokens fill in details where excessive exploration can harm coherence.

**Our Strategy**: *Explore at the beginning, exploit at the end*

EAD implements an intuitive temperature annealing schedule that:
- Starts with high temperature (τ > 1) to encourage diverse exploration of solution paths
- Gradually cools to lower temperatures to ensure coherent, high-quality completions  
- Maintains proximity to the target policy for stable off-policy learning

## Mathematical Formulation

EAD uses a dynamic temperature schedule that starts high and gradually decreases:

$$\tau_t = \max\{1 + \tau_\mathrm{max} - e^{t/d}, \tau_\mathrm{min}\}$$

Where:
- $\tau_t$ is the temperature at token position $t$
- $\tau_\mathrm{max} > 1$ is the maximum temperature for exploration
- $\tau_\mathrm{min}$ is the minimum temperature for exploitation
- $d$ is the decay rate controlling annealing speed

The decay rate is made global-step-aware to adapt to increasing response lengths:

$$d_s = \min(d_0 + 5s, 40000)$$

Where $s$ is the training step, ensuring the annealing schedule scales with model capabilities.

## Key Benefits

- **Plug-and-Play Enhancement**: Improves sample efficiency over fixed-temperature sampling
- **Broad Compatibility**: Works with various RLVR algorithms (GRPO, DAPO, EntropyMech)
- **Sample Efficient**: Achieves strong results with fewer rollouts
- **Inference-Time Benefits**: Also improves generation quality at test time
- **Mitigates Entropy Collapse**: Helps escape local optima during training plateaus

## Key Results

> **Note**: For the best viewing experience, please visit our [research website](https://yangalan123.github.io/ead_rlvr/) to see all figures in high resolution.

**Figure 1: Annealing Schedule**
- Shows how different decay rates d affect the temperature schedule
- A larger d slows the cooling, front-loading exploration over more tokens
- [View full figure](figures/annealing_schedule_plot.pdf)

**Figure 2: Performance Results** 
- Pass@16 and Worst@16 evaluation in RL training
- EAD significantly improves exploration of high-quality samples
- [View full figure](figures/best-and-worst-at-16.pdf)

**Figure 3: Entropy Dynamics**
- EAD mitigates entropy collapse by maintaining exploration throughout training
- Helps escape local optima during plateau stages
- [View full figure](figures/entropy.pdf)

**Figure 4: Algorithm Compatibility**
- EAD works with various RL algorithms (GRPO, EntropyMech)
- Consistently outperforms fixed-temperature sampling
- [View full figure](figures/diff-method.pdf)

## Try EAD
Check out the implementation in `recipe/ead/` and explore the research [website](https://yangalan123.github.io/ead_rlvr/) for detailed results and visualizations.

**Quick Start with EAD:**
```bash
# Follow Minimal-RL to prepare the dataset
# Then run EAD with annealed sampling
cd recipe/ead
bash run_annealed_sampling.sh
```