#!/usr/bin/env python3
"""
exp_006 — mHC Stream Simulation

Testar om inter-ström-divergens i en simulerad mHC-arkitektur
korrelerar med känd epistemisk osäkerhet (Ue).

Hypotesen (HYPOTHESIS.md §2, Signal 1):
    Hög divergens mellan n parallella strömmar → hög U_model

Ingen riktig mHC-modell behövs. Vi simulerar strömbeteende syntetiskt
och mäter om signalen är meningsfull.

Körning:
    python simulate_mhc_streams.py

Beroenden: numpy, scipy (standardbibliotek)
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr


# ── Sinkhorn-Knopp ──────────────────────────────────────────────────────────

def sinkhorn_knopp(M: np.ndarray, n_iter: int = 20) -> np.ndarray:
    """Projicerar M på Birkhoff-polytopen (dubbelt stokastisk matris)."""
    M = np.exp(M - M.max())  # numerisk stabilitet
    for _ in range(n_iter):
        M = M / M.sum(axis=1, keepdims=True)
        M = M / M.sum(axis=0, keepdims=True)
    return M


# ── Mätfunktioner ───────────────────────────────────────────────────────────

def inter_stream_divergence(streams: np.ndarray) -> float:
    """
    Genomsnittlig L2-avstånd från varje ström till medelvektorn.
    streams: (n_streams, d)
    """
    mean = streams.mean(axis=0)
    return float(np.mean([np.linalg.norm(s - mean) for s in streams]))


def mixing_entropy(H_res: np.ndarray) -> float:
    """Entropi i H_res-matrisen. Hög = diffus blandning, låg = ordnad."""
    H = np.clip(H_res, 1e-10, 1.0)
    return float(-np.sum(H * np.log(H)))


def routing_concentration(H_pre: np.ndarray) -> float:
    """
    Gini-koefficient för H_pre (routingvektor).
    Hög Gini = en ström dominerar = koncentrerad routing.
    """
    v = np.sort(np.abs(H_pre.flatten()))
    n = len(v)
    cumsum = np.cumsum(v)
    return float((2 * np.sum(cumsum) - (n + 1) * v.sum()) / (n * v.sum() + 1e-10))


# ── Simulering ───────────────────────────────────────────────────────────────

def simulate_layer(true_ue: float, n_streams: int = 4, d: int = 64,
                   seed: int = 0) -> dict:
    """
    Simulerar ett mHC-lager med känd osäkerhet (true_ue ∈ [0, 1]).

    Låg Ue  → strömmar nära varandra (liten spridning)
    Hög Ue  → strömmar divergerar (stor spridning)
    """
    rng = np.random.default_rng(seed)

    # Basrepresentation för lagret
    base = rng.standard_normal(d)

    # n parallella strömmar — brus skalas med true_ue
    noise_scale = true_ue * 3.0
    streams = np.stack([base + rng.standard_normal(d) * noise_scale
                        for _ in range(n_streams)])

    # H_res: dubbelt stokastisk via Sinkhorn-Knopp
    H_res_raw = rng.standard_normal((n_streams, n_streams))
    H_res = sinkhorn_knopp(H_res_raw)

    # H_pre: routingvektor (aggregering till layer input)
    # Låg Ue → koncentrerad, Hög Ue → flat
    h_pre_raw = rng.standard_normal(n_streams)
    if true_ue < 0.3:
        # Koncentrera mot en ström
        h_pre_raw[0] += 3.0
    H_pre = np.exp(h_pre_raw) / np.exp(h_pre_raw).sum()

    return {
        "true_ue": true_ue,
        "divergence": inter_stream_divergence(streams),
        "entropy": mixing_entropy(H_res),
        "gini": routing_concentration(H_pre),
        "streams": streams,
        "H_res": H_res,
        "H_pre": H_pre,
    }


# ── Experiment ───────────────────────────────────────────────────────────────

def run_experiment(n_samples: int = 200, n_streams: int = 4, d: int = 64):
    """
    Kör simulering över ett spektrum av true_ue-värden.
    Mäter korrelation mellan sann Ue och varje signal.
    """
    rng = np.random.default_rng(42)
    ue_values = rng.uniform(0.0, 1.0, n_samples)

    results = [simulate_layer(ue, n_streams=n_streams, d=d, seed=i)
               for i, ue in enumerate(ue_values)]

    true_ues = np.array([r["true_ue"] for r in results])
    divergences = np.array([r["divergence"] for r in results])
    entropies = np.array([r["entropy"] for r in results])
    ginis = np.array([r["gini"] for r in results])

    # Korrelationsanalys
    div_pearson, div_p = pearsonr(true_ues, divergences)
    ent_pearson, ent_p = pearsonr(true_ues, entropies)
    gini_pearson, gini_p = pearsonr(true_ues, ginis)

    div_spearman, _ = spearmanr(true_ues, divergences)
    ent_spearman, _ = spearmanr(true_ues, entropies)
    gini_spearman, _ = spearmanr(true_ues, ginis)

    print("=" * 60)
    print("exp_006 — mHC Stream Simulation")
    print(f"n_samples={n_samples}, n_streams={n_streams}, d={d}")
    print("=" * 60)
    print()
    print("KORRELATION MED KÄND Ue (true_ue)")
    print("-" * 60)
    print(f"{'Signal':<30} {'Pearson r':>10} {'p-värde':>10} {'Spearman':>10}")
    print("-" * 60)
    print(f"{'Signal 1: inter-ström-divergens':<30} {div_pearson:>10.3f} {div_p:>10.4f} {div_spearman:>10.3f}")
    print(f"{'Signal 2: H_res mixing-entropi':<30} {ent_pearson:>10.3f} {ent_p:>10.4f} {ent_spearman:>10.3f}")
    print(f"{'Signal 3: routing Gini (inv.)':<30} {-gini_pearson:>10.3f} {gini_p:>10.4f} {-gini_spearman:>10.3f}")
    print()

    # Tolkning
    print("TOLKNING")
    print("-" * 60)
    signals = [
        ("Signal 1 (divergens)", div_pearson, div_p),
        ("Signal 2 (entropi)", ent_pearson, ent_p),
        ("Signal 3 (Gini inv.)", -gini_pearson, gini_p),
    ]
    for name, r, p in signals:
        strength = "stark" if abs(r) > 0.7 else "måttlig" if abs(r) > 0.4 else "svag"
        sig = "signifikant" if p < 0.05 else "EJ signifikant"
        print(f"  {name}: r={r:.3f} — {strength}, {sig}")

    print()
    print("SLUTSATS")
    print("-" * 60)
    strong_signals = sum(1 for _, r, p in signals if abs(r) > 0.4 and p < 0.05)
    if strong_signals >= 2:
        print("  ✓ Minst 2 signaler korrelerar med Ue.")
        print("    Hypotesen är värd empirisk validering på riktiga modeller.")
    elif strong_signals == 1:
        print("  ~ En signal korrelerar. Partiellt stöd.")
        print("    Behöver justeras innan vidare experiment.")
    else:
        print("  ✗ Ingen stark korrelation funnen.")
        print("    Hypotesen i denna form stöds inte av simuleringen.")

    return {
        "true_ues": true_ues,
        "divergences": divergences,
        "entropies": entropies,
        "ginis": ginis,
        "correlations": {
            "divergence": (div_pearson, div_p),
            "entropy": (ent_pearson, ent_p),
            "gini": (-gini_pearson, gini_p),
        }
    }


if __name__ == "__main__":
    run_experiment()
