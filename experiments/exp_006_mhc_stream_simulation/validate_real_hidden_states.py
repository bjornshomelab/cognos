#!/usr/bin/env python3
"""
exp_006b — Validering mot riktiga hidden states (GPT-2)

Testar om cross-head divergens i multi-head attention korrelerar
med logit-entropi (proxy för epistemisk osäkerhet).

Proxy-logik:
  mHC n parallella strömmar  ≈  transformer attention-huvuden
  inter-ström-divergens       ≈  cross-head divergens i MHA

Prompter med känd osäkerhet:
  - Faktafrågor (låg Ue förväntas)
  - Tvetydiga frågor (hög Ue förväntas)
  - Nonsens-input (maximal Ue)

Körning:
    python3 validate_real_hidden_states.py
"""

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model

# ── Prompts med känd förväntad osäkerhet ────────────────────────────────────

PROMPTS = [
    # (text, förväntad_ue_nivå, beskrivning)
    ("The capital of France is", "low", "faktafråga — Paris"),
    ("The capital of Sweden is", "low", "faktafråga — Stockholm"),
    ("Water boils at", "low", "faktafråga — 100C"),
    ("The president of the United States is", "medium", "faktafråga med tidsberoende"),
    ("The best programming language is", "high", "subjektiv — ingen sanning"),
    ("Whether God exists is", "high", "filosofisk — genuint oklar"),
    ("xkqz blorb fntt mwp", "high", "nonsens — maximal osäkerhet"),
    ("The meaning of life is", "high", "öppen fråga"),
    ("2 + 2 =", "low", "aritmetik — trivial"),
    ("The most ethical action in a trolley problem is", "high", "etisk dilemma"),
]

# ── Mätfunktioner ────────────────────────────────────────────────────────────

def cross_head_divergence(head_outputs: torch.Tensor) -> float:
    """
    Genomsnittlig L2-divergens mellan attention-huvuden.
    head_outputs: (n_heads, seq_len, head_dim)
    """
    # Medelvektorn för varje huvud (kollapsa seq_len → (n_heads, head_dim))
    head_means = head_outputs.mean(dim=1)  # (n_heads, head_dim)
    global_mean = head_means.mean(dim=0)   # (head_dim,)
    divergences = torch.norm(head_means - global_mean, dim=1)  # (n_heads,)
    return float(divergences.mean().item())


def logit_entropy_proxy(model, tokenizer, text: str) -> float:
    """
    Approximerar Ue via varians i näst-sista lagrets hidden state-norm.
    (Utan LM head — vi mäter intern spridning, inte output-sannolikheter)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Varians i sista hidden state som Ue-proxy
    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
    return float(last_hidden.var().item())


# ── Huvudexperiment ───────────────────────────────────────────────────────────

def run_experiment():
    print("Laddar GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_attentions=True)
    model.eval()

    n_heads = model.config.n_head          # 12 i GPT-2 small
    head_dim = model.config.n_embd // n_heads  # 64

    print(f"Modell: GPT-2 small | n_heads={n_heads} | head_dim={head_dim}")
    print()

    results = []

    for text, expected_ue, desc in PROMPTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

        # Samla cross-head divergens per lager
        layer_divergences = []
        for layer_attn in outputs.attentions:
            # layer_attn: (batch, n_heads, seq_len, seq_len) — attention weights
            # Vi vill ha per-huvud representationer — använd hidden states istället
            pass

        # Bättre: extrahera per-huvud hidden states via omformning
        # hidden_states[-1]: (1, seq_len, n_embd)
        # Dela upp i n_heads × head_dim
        last_hidden = outputs.hidden_states[-1]  # (1, seq_len, n_embd)
        seq_len = last_hidden.shape[1]

        # Omforma till (n_heads, seq_len, head_dim)
        head_repr = last_hidden.view(1, seq_len, n_heads, head_dim)
        head_repr = head_repr.squeeze(0).permute(1, 0, 2)  # (n_heads, seq_len, head_dim)

        divergence = cross_head_divergence(head_repr)
        ue_proxy = logit_entropy_proxy(model, tokenizer, text)

        results.append({
            "text": text[:40],
            "desc": desc,
            "expected": expected_ue,
            "divergence": divergence,
            "ue_proxy": ue_proxy,
        })

    # ── Resultat ──────────────────────────────────────────────────────────────
    print("=" * 72)
    print("exp_006b — Cross-head divergens vs Ue-proxy (GPT-2 small)")
    print("=" * 72)
    print(f"{'Prompt':<42} {'Exp':>5} {'Div':>8} {'Ue-proxy':>10}")
    print("-" * 72)

    low_divs, high_divs = [], []
    for r in results:
        marker = "✓" if (
            (r["expected"] == "low" and r["divergence"] < np.median([x["divergence"] for x in results])) or
            (r["expected"] == "high" and r["divergence"] >= np.median([x["divergence"] for x in results]))
        ) else " "
        print(f"{marker} {r['text']:<40} {r['expected']:>5} {r['divergence']:>8.4f} {r['ue_proxy']:>10.6f}")
        if r["expected"] == "low":
            low_divs.append(r["divergence"])
        elif r["expected"] == "high":
            high_divs.append(r["divergence"])

    print()
    low_mean = np.mean(low_divs)
    high_mean = np.mean(high_divs)
    ratio = high_mean / low_mean if low_mean > 0 else float("inf")

    print(f"Genomsnittlig divergens — låg Ue: {low_mean:.4f}")
    print(f"Genomsnittlig divergens — hög Ue: {high_mean:.4f}")
    print(f"Ratio hög/låg: {ratio:.2f}x")
    print()

    print("SLUTSATS")
    print("-" * 72)
    if ratio > 1.5:
        print(f"  ✓ Cross-head divergens är {ratio:.1f}x högre för osäkra prompts.")
        print("    Signalen är meningsfull i riktiga transformer-internals.")
        print("    Hypotes 2 stöds — värd vidare validering på mHC-modeller.")
    elif ratio > 1.1:
        print(f"  ~ Svagt stöd (ratio={ratio:.2f}x). Signalen finns men är brusig.")
        print("    Kan bero på att GPT-2 är för liten / prompt-set för litet.")
    else:
        print(f"  ✗ Ingen klar skillnad (ratio={ratio:.2f}x).")
        print("    Cross-head divergens i standard MHA ≠ inter-ström-divergens i mHC.")
        print("    Hypotesen kräver faktiska mHC-modeller för validering.")


if __name__ == "__main__":
    run_experiment()
