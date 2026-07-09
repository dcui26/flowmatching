# Flow Matching and Rectified Flow on CIFAR-10

An implementation of Conditional Flow Matching (CFM) and Rectified Flow (Reflow) for unconditional image generation on CIFAR-10. Compares the two methods across different amounts of simulation steps from 1 to 100, showing the tradeoff between peak sample quality and few-step generation efficiency

## Overview

Flow matching learns a velocity field that transports noise to data via an ODE. At inference time, the ODE is simulated to obtain a sample from noise, requiring an approximation of the trajectory due to integration across time being intractable. Rectified flow (reflow) straightens the learned trajectories by training a second model on (noise, generated image) couplings from the first, reducing the integration approximation error and enabling accurate generation in far fewer steps.

## Architecture

The backbone is a DiT-S/2 style transformer (Peebles & Xie, 2023) with patchification and adaLN-Zero conditioning, adapted for flow matching:

- Patch size: 2×2
- Hidden dim: 384, depth: 12, heads: 6
- Parameters: 32.47M
- Unconditional (no class labels)

## Training

**CFM**: ~195K steps on CIFAR-10 training set. AdamW with lr=2e-4, no weight decay, batch size 128, constant LR schedule with 2500-step linear warmup. EMA decay 0.9999 with 5-epoch warmup.

**Reflow**: ~117K steps on 50K (x₀, x₁) couplings extracted from the CFM EMA model using 100-step Heun sampling. Same optimizer and architecture, trained from scratch.

## Results

All evaluations use 50K generated samples, Heun sampling, and FID computed against the CIFAR-10 training set using Inception-v3 features.

![FID vs NFE]

| Model | Steps | NFE | FID ↓ | Precision ↑ | Recall ↑ |
|-------|-------|-----|-------|-------------|----------|
| CFM | 100 | 200 | **7.86** | 0.7165 | 0.6766 |
| CFM | 50 | 100 | 8.59 | 0.7050 | 0.6669 |
| CFM | 25 | 50 | 14.51 | 0.6831 | 0.6169 |
| CFM | 10 | 20 | 48.18 | 0.5174 | 0.3983 |
| CFM | 5 | 10 | 119.10 | 0.4318 | 0.1386 |
| CFM | 2 | 4 | 333.08 | 0.6197 | 0.0003 |
| CFM | 1 | 2 | 476.80 | 0.0000 | 0.0000 |
| Reflow | 100 | 200 | 11.92 | 0.7115 | 0.6381 |
| Reflow | 50 | 100 | 12.01 | 0.7087 | 0.6358 |
| Reflow | 25 | 50 | 11.91 | 0.7136 | 0.6315 |
| Reflow | 10 | 20 | **12.32** | 0.7117 | 0.6361 |
| Reflow | 5 | 10 | 16.86 | 0.6868 | 0.5922 |
| Reflow | 2 | 4 | 63.28 | 0.4745 | 0.2839 |
| Reflow | 1 | 2 | 169.06 | 0.2357 | 0.0077 |

### Key findings

**Reflow is nearly flat from 10 to 100 steps.** FID stays within 11.9–12.3 across this entire range, meaning you can reduce the step count by 10x with essentially no quality loss.

**CFM degrades rapidly below 50 steps.** FID explodes from 8.59 at 50 steps to 48.18 at 10 steps and 476.80 at 1 step. The velocity field has too much curvature for large integration steps.

**The crossover occurs at ~25 steps (50 NFE).** Below this point, reflow outperforms CFM at matched compute. At 10 steps, the gap is 4x (12.32 vs 48.18).

**Reflow trades peak quality for robustness.** At high step counts, CFM achieves better peak quality (FID 7.86 vs 11.92) since the reflow model trains on generated samples rather than real data, inheriting imperfections from the parent model.

**Single-step generation requires multiple reflow rounds.** One round of reflow is not sufficient for 1-step generation (FID 169). The literature shows 2–3 successive rounds of reflow are needed to achieve true single-step quality.

## File structure

- `cfm.py` — CFM loss, Euler sampling, Heun sampling
- `rectify.py` — Reflow loss, coupling extraction, coupled dataset
- `transformer_backbone.py` — DiT-S/2 FlowTransformer with adaLN-Zero
- `train_cfm.py` — CFM training loop with EMA and checkpointing
- `train_rectify.py` — Reflow training loop
- `extract_coupling.py` — Generate and save (x₀, x₁) couplings from a trained CFM model
- `eval.py` — FID, precision, and recall evaluation

## References

- Lipman et al., "Flow Matching for Generative Modeling" (2022). [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (2022). [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023). [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
