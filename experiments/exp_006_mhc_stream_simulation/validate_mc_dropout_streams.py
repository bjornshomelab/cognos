#!/usr/bin/env python3
"""
exp_006c — MC-Dropout som mHC-strömsimulering

MC-dropout skapar n stokastiska varianter av samma representation.
Det är konceptuellt närmre mHC än attention-huvuden:
  - mHC:   n deterministiska men separata strömmar per lager
  - MC-DO: n stokastiska körningar = n "strömmar" via dropout-masker

Körning:
    python3 validate_mc_dropout_streams.py
"""

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from scipy.stats import pearsonr, spearmanr

N_STREAMS = 8   # antal MC-dropout-körningar per prompt
DROPOUT_P = 0.1  # dropout-sannolikhet

PROMPTS = [
    ("The capital of France is", "low",    "fakta"),
    ("The capital of Sweden is", "low",    "fakta"),
    ("Water boils at 100",        "low",    "fakta"),
    ("2 + 2 =",                   "low",    "aritmetik"),
    ("The best programming language is", "high", "subjektiv"),
    ("Whether God exists is",     "high",   "filosofisk"),
    ("xkqz blorb fntt mwp",       "high",   "nonsens"),
    ("The meaning of life is",    "high",   "öppen"),
    ("The most ethical action in a trolley problem is", "high", "etik"),
    ("The president in 1850 was", "medium", "historisk/tidsberoende"),
]


def enable_dropout(model):
    """Aktiverar dropout-lager för MC-sampling."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def mc_stream_divergence(model, tokenizer, text: str, n: int = N_STREAMS) -> float:
    """
    Kör forward pass n gånger med dropout aktiverat.
    Returnerar genomsnittlig L2-divergens mellan körningarna.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)

    # Aktivera dropout för stokastiska körningar
    model.eval()
    enable_dropout(model)

    streams = []
    with torch.no_grad():
        for _ in range(n):
            out = model(**inputs, output_hidden_states=True)
            # Sista hidden state, medelvärde över tokens → (hidden_dim,)
            h = out.hidden_states[-1].squeeze(0).mean(dim=0)
            streams.append(h)

    streams = torch.stack(streams)  # (n, hidden_dim)
    mean = streams.mean(dim=0)
    divergences = torch.norm(streams - mean, dim=1)
    return float(divergences.mean().item())


def run_experiment():
    print("Laddar GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")

    print(f"n_streams={N_STREAMS}, dropout_p={DROPOUT_P}")
    print()

    results = []
    for text, expected, desc in PROMPTS:
        div = mc_stream_divergence(model, tokenizer, text)
        results.append({
            "text": text[:45],
            "desc": desc,
            "expected": expected,
            "divergence": div,
        })

    median_div = np.median([r["divergence"] for r in results])

    print("=" * 68)
    print("exp_006c — MC-Dropout strömdivergens vs känd Ue")
    print("=" * 68)
    print(f"{'Prompt':<47} {'Exp':>6} {'Div':>9}")
    print("-" * 68)

    low_divs, high_divs = [], []
    for r in results:
        correct = (
            (r["expected"] == "low"  and r["divergence"] < median_div) or
            (r["expected"] == "high" and r["divergence"] >= median_div)
        )
        marker = "✓" if correct else " "
        print(f"{marker} {r['text']:<46} {r['expected']:>6} {r['divergence']:>9.4f}")
        if r["expected"] == "low":
            low_divs.append(r["divergence"])
        elif r["expected"] == "high":
            high_divs.append(r["divergence"])

    low_mean  = np.mean(low_divs)
    high_mean = np.mean(high_divs)
    ratio = high_mean / low_mean if low_mean > 0 else float("inf")

    # Spearman-korrelation mot förväntad Ue (kodad som 0=low, 0.5=medium, 1=high)
    expected_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
    expected_vals = [expected_map[r["expected"]] for r in results]
    divs          = [r["divergence"] for r in results]
    spearman_r, spearman_p = spearmanr(expected_vals, divs)

    print()
    print(f"Genomsnittlig div — låg Ue:  {low_mean:.4f}")
    print(f"Genomsnittlig div — hög Ue:  {high_mean:.4f}")
    print(f"Ratio hög/låg:               {ratio:.2f}x")
    print(f"Spearman r:                  {spearman_r:.3f}  (p={spearman_p:.4f})")
    print()

    print("SLUTSATS")
    print("-" * 68)
    if ratio > 1.5 and spearman_r > 0.4:
        print(f"  ✓ Stark signal: ratio={ratio:.2f}x, Spearman r={spearman_r:.3f}")
        print("    MC-dropout strömmar approximerar mHC-signalen väl.")
        print("    Hypotes 2 stöds — MC-dropout kan användas som interim-validering.")
    elif ratio > 1.2 or spearman_r > 0.3:
        print(f"  ~ Måttlig signal: ratio={ratio:.2f}x, Spearman r={spearman_r:.3f}")
        print("    MC-dropout fungerar som proxy men med brus.")
        print("    Behöver fler prompts eller högre n_streams.")
    else:
        print(f"  ✗ Svag signal: ratio={ratio:.2f}x, Spearman r={spearman_r:.3f}")
        print("    MC-dropout i GPT-2 ger inte tillräcklig separation.")
        print("    Gå vidare med alternativ 3 (teoretiskt bidrag).")


if __name__ == "__main__":
    run_experiment()
