# exp_006 — Findings: mHC Stream Simulation

**Date:** 2026-02-23
**Researcher:** Björn Wikström

---

## Question

Can inter-stream divergence in an mHC-like architecture serve as a native epistemic signal for CognOS?

## Three Experiments

### exp_006a — Synthetic simulation

- n=200 samples, n_streams=4, d=64
- Noise scaled directly with true_ue
- **Signal 1 (inter-stream divergence): Pearson r = 0.996, p < 0.0001** ✓
- Signal 3 (Gini routing concentration): r = 0.48, significant ✓
- Signal 2 (H_res mixing entropy): r = -0.02, not significant ✗

### exp_006b — GPT-2, attention heads as proxy

- Final hidden state split into 12 heads (n_heads × head_dim)
- Ratio high/low Ue: **1.11x** — weak support
- Limitation: reshaping destroyed per-head information

### exp_006c — GPT-2, MC-dropout as proxy

- n=8 stochastic forward passes per prompt (dropout_p=0.1)
- Ratio high/low Ue: **0.87x** — no signal
- Spearman r = 0.13, not significant

## Conclusion

The signal exists **mathematically and synthetically** (r=0.996), but **cannot be approximated** in standard GPT-2 via attention heads or MC-dropout. Both proxy methods introduce too much noise.

**Reason:** mHC streams are deterministically separated via a learned H_res matrix. Standard transformers have no equivalent — the streams must exist architecturally.

## Implication for CognOS

The hypothesis is strong enough for a theoretical contribution / position paper section. Full empirical validation awaits:

- Public mHC model weights from DeepSeek or community
- Alternatively: micro-mHC implementation trained from scratch

## Connections

- **HYPOTHESIS.md §2** — Signal 1–3 formally defined
- **INSIGHTS.md Insight 9** — quantum gate analogy and pre-collapse measurement window
- **Naimat Ullah (LinkedIn)** — his "Intervention State" UI is designed to catch exactly this signal
