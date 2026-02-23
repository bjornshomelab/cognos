#!/usr/bin/env python3
"""
exp_007 — MNIST som kontrollerad testbädd för CognOS-signaler

Fråga: Korrelerar hidden state-divergens med softmax-entropi (validerad Ue)?

MNIST ger oss något GPT-2 inte hade:
  - Ground truth per sample
  - Softmax-entropi som kalibrerad Ue-referens
  - Kända svåra siffror (4,7,9) vs enkla (0,1)

Två experiment:
  exp_007a — Statisk strömdelning (dense(128) → 4 strömmar à 32)
  exp_007b — MC-dropout strömmar (n=10 körningar per sample)

Körning:
    python3 run_mnist_streams.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

# ── Konfiguration ─────────────────────────────────────────────────────────────

N_STREAMS   = 4      # virtuella strömmar (128 / 4 = 32 per ström)
N_MC        = 10     # MC-dropout körningar
DROPOUT_P   = 0.2
BATCH_SIZE  = 256
EPOCHS      = 5
EVAL_N      = 2000   # antal testsamples att analysera

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Siffror med känd svårighetsgrad
EASY_DIGITS = {0, 1}
HARD_DIGITS = {4, 7, 9}


# ── Modell ────────────────────────────────────────────────────────────────────

class MNISTNet(nn.Module):
    def __init__(self, dropout_p=DROPOUT_P):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(28 * 28, 128)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x, return_hidden=False):
        x = self.flatten(x)
        h = F.relu(self.fc1(x))       # (batch, 128) — hidden state
        d = self.dropout(h)
        out = self.fc2(d)             # (batch, 10) logits
        if return_hidden:
            return out, h
        return out


# ── Träning ───────────────────────────────────────────────────────────────────

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


# ── Signalmätning ─────────────────────────────────────────────────────────────

def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """H(p) = -sum(p * log(p+eps)), ett per sample."""
    probs = F.softmax(logits, dim=1)
    return -(probs * (probs + 1e-9).log()).sum(dim=1)


def static_stream_divergence(hidden: torch.Tensor, n_streams: int) -> torch.Tensor:
    """
    Dela hidden state i n_streams virtuella strömmar.
    hidden: (batch, 128)
    Returnerar genomsnittlig L2-divergens mellan strömmar per sample.
    """
    batch, dim = hidden.shape
    stream_dim = dim // n_streams
    streams = hidden.view(batch, n_streams, stream_dim)   # (batch, n, d)
    mean    = streams.mean(dim=1, keepdim=True)            # (batch, 1, d)
    div     = torch.norm(streams - mean, dim=2).mean(dim=1)  # (batch,)
    return div


def mc_stream_divergence(model, x: torch.Tensor, n: int) -> torch.Tensor:
    """
    n MC-dropout körningar → n 'strömmar'.
    Returnerar divergens per sample.
    """
    model.eval()
    # Aktivera dropout
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    runs = []
    with torch.no_grad():
        for _ in range(n):
            _, h = model(x, return_hidden=True)
            runs.append(h)                          # (batch, 128)

    stacked = torch.stack(runs, dim=1)              # (batch, n, 128)
    mean    = stacked.mean(dim=1, keepdim=True)
    div     = torch.norm(stacked - mean, dim=2).mean(dim=1)
    return div


# ── Huvudexperiment ───────────────────────────────────────────────────────────

def run():
    print(f"Enhet: {DEVICE}")

    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST("/tmp/mnist", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("/tmp/mnist", train=False, download=True, transform=transform)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # Träning
    model     = MNISTNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    print(f"\nTränar MNIST ({EPOCHS} epoker)...")
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_ld, optimizer, criterion)
        acc  = evaluate(model, test_ld)
        print(f"  Epok {epoch}: loss={loss:.4f}  test-acc={acc:.4f}")

    # Samla testdata (EVAL_N samples)
    model.eval()
    all_x, all_y = [], []
    for x, y in test_ld:
        all_x.append(x)
        all_y.append(y)
        if sum(len(b) for b in all_x) >= EVAL_N:
            break
    all_x = torch.cat(all_x)[:EVAL_N].to(DEVICE)
    all_y = torch.cat(all_y)[:EVAL_N].numpy()

    # Mät signaler
    with torch.no_grad():
        logits, hidden = model(all_x, return_hidden=True)

    entropy   = softmax_entropy(logits).cpu().numpy()
    static_div = static_stream_divergence(hidden, N_STREAMS).cpu().numpy()
    mc_div    = mc_stream_divergence(model, all_x, N_MC).cpu().numpy()

    # ── Resultat exp_007a ─────────────────────────────────────────────────────
    r_static, p_static = pearsonr(entropy, static_div)
    rs_static, ps_static = spearmanr(entropy, static_div)

    print(f"\n{'='*62}")
    print("exp_007a — Statisk strömdelning (4 × 32) vs softmax-entropi")
    print(f"{'='*62}")
    print(f"Pearson  r = {r_static:.3f}  (p={p_static:.4f})")
    print(f"Spearman r = {rs_static:.3f}  (p={ps_static:.4f})")

    # Svåra vs enkla siffror
    easy_mask = np.isin(all_y, list(EASY_DIGITS))
    hard_mask = np.isin(all_y, list(HARD_DIGITS))
    ratio_static = static_div[hard_mask].mean() / static_div[easy_mask].mean()
    print(f"Ratio hard/easy: {ratio_static:.2f}x  "
          f"(hard={static_div[hard_mask].mean():.4f}, easy={static_div[easy_mask].mean():.4f})")

    if abs(r_static) > 0.5:
        print("  ✓ Stark korrelation — statisk strömdelning fångar Ue-signal")
    elif abs(r_static) > 0.3:
        print("  ~ Måttlig korrelation")
    else:
        print("  ✗ Svag korrelation — arbiträr delning bär ingen signal")

    # ── Resultat exp_007b ─────────────────────────────────────────────────────
    r_mc, p_mc = pearsonr(entropy, mc_div)
    rs_mc, ps_mc = spearmanr(entropy, mc_div)
    ratio_mc = mc_div[hard_mask].mean() / mc_div[easy_mask].mean()

    print(f"\n{'='*62}")
    print(f"exp_007b — MC-dropout ({N_MC} körningar) vs softmax-entropi")
    print(f"{'='*62}")
    print(f"Pearson  r = {r_mc:.3f}  (p={p_mc:.4f})")
    print(f"Spearman r = {rs_mc:.3f}  (p={ps_mc:.4f})")
    print(f"Ratio hard/easy: {ratio_mc:.2f}x  "
          f"(hard={mc_div[hard_mask].mean():.4f}, easy={mc_div[easy_mask].mean():.4f})")

    if abs(r_mc) > 0.5:
        print("  ✓ Stark korrelation — MC-dropout fångar Ue-signal")
    elif abs(r_mc) > 0.3:
        print("  ~ Måttlig korrelation")
    else:
        print("  ✗ Svag korrelation — MC-dropout bär ingen signal här")

    # ── Jämförelse ────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("SLUTSATS")
    print(f"{'='*62}")
    print(f"  Statisk delning:  Pearson r={r_static:.3f}, ratio={ratio_static:.2f}x")
    print(f"  MC-dropout:       Pearson r={r_mc:.3f}, ratio={ratio_mc:.2f}x")

    winner = "MC-dropout" if abs(r_mc) > abs(r_static) else "Statisk delning"
    print(f"\n  Starkare signal: {winner}")

    if abs(r_mc) > 0.5 or abs(r_static) > 0.5:
        print("  → Hidden state-divergens korrelerar med Ue i kontrollerad miljö.")
        print("    Stödjer att signalen FINNS i riktiga modeller — men kräver")
        print("    arkitekturellt separerade strömmar (mHC) för maximal styrka.")
    else:
        print("  → Divergensen bär ingen signal utan arkitekturell separation.")
        print("    Bekräftar exp_006-slutsatsen: mHC-internals krävs.")


if __name__ == "__main__":
    run()
