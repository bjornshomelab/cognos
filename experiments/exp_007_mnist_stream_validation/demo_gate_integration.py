#!/usr/bin/env python3
"""
Demo: CognOS UncertaintyGate integrerad med MNIST-modellen

Visar hur gaten sitter vid varje reasoning-steg:

    Input → [Dense 128 + ReLU] → [Gate 1: hidden state] →
             [Dense 10 Softmax]  → [Gate 2: logits]      → beslut

Gate 1 mäter divergens i hidden state (stream proxy)
Gate 2 mäter entropi i logits (validerad Ue)

Körning:
    python3 demo_gate_integration.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from cognos.gate import UncertaintyGate, ReasoningPipeline

# ── Modell (samma som exp_007) ────────────────────────────────────────────────

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(28 * 28, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        h = F.relu(self.fc1(x))    # hidden state (128,)
        d = self.dropout(h)
        logits = self.fc2(d)       # logits (10,)
        return logits, h


# ── Bygg och träna (snabb) ────────────────────────────────────────────────────

def train_model():
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST("/tmp/mnist", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("/tmp/mnist", train=False, download=True, transform=transform)
    train_ld = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_ld  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    model     = MNISTNet()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    print("Tränar MNIST (5 epoker)...")
    for epoch in range(1, 6):
        model.train()
        for x, y in train_ld:
            optimizer.zero_grad()
            logits, _ = model(x)
            criterion(logits, y).backward()
            optimizer.step()

    # Testprecision
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_ld:
            logits, _ = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total   += len(y)
    print(f"Test-accuracy: {correct/total:.4f}\n")
    return model


# ── Gate-integration ──────────────────────────────────────────────────────────

def run_with_gates(model):
    """
    Kör MNIST-inferens med CognOS-gates vid varje reasoning-steg.

    Gate 1 — hidden state (variance proxy, inverterat tecken utan mHC)
    Gate 2 — logits (normaliserad softmax-entropi, validerad signal)
    """

    # Gate 1: behandla hidden state som unnormaliserade logits — grov proxy.
    # Utan mHC är detta lenient (hög high_cutoff). Med mHC-weights: sänk till 0.40.
    gate_hidden = UncertaintyGate(low=0.10, high=0.90, source="entropy")

    # Gate 2: logit-entropi — validerad Ue-signal (exp_007, positiv korrelation)
    gate_logits = UncertaintyGate(low=0.10, high=0.35, source="entropy")

    pipeline = ReasoningPipeline([gate_hidden, gate_logits])

    # Exempelprompter — handplockade siffror
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds   = datasets.MNIST("/tmp/mnist", train=False, download=True, transform=transform)

    # Välj 2 samples per klass (0-9)
    samples_per_class = {i: [] for i in range(10)}
    for img, label in test_ds:
        lbl = int(label)
        if len(samples_per_class[lbl]) < 2:
            samples_per_class[lbl].append((img, lbl))
        if all(len(v) >= 2 for v in samples_per_class.values()):
            break

    print("=" * 62)
    print("CognOS Gate-integration på MNIST")
    print("Gate 1 (hidden, variance) | Gate 2 (logits, entropy)")
    print("=" * 62)
    print(f"{'Siffra':>6} {'Pred':>5} {'G1-Ue':>7} {'G1':>9} {'G2-Ue':>7} {'G2':>9} {'Utfall':>9}")
    print("-" * 62)

    decision_counts = {"pass": 0, "explore": 0, "escalate": 0}

    model.eval()
    with torch.no_grad():
        for digit in range(10):
            for img, label in samples_per_class[digit]:
                logits, hidden = model(img.unsqueeze(0))
                logits_np = logits.squeeze(0).numpy()
                hidden_np = hidden.squeeze(0).numpy()
                pred = int(logits_np.argmax())

                result = pipeline.run([hidden_np, logits_np])
                g1 = result.history[0]
                g2 = result.history[1] if len(result.history) > 1 else None

                g2_ue  = f"{g2.ue:.3f}" if g2 else "  —  "
                g2_dec = g2.decision if g2 else "skipped"
                outcome = result.outcome
                decision_counts[outcome] += 1

                correct = "✓" if pred == label else "✗"
                print(f"  {label:>4}  {correct}{pred:>3}  "
                      f"{g1.ue:>7.3f} {g1.decision:>9}  "
                      f"{g2_ue:>7} {g2_dec:>9}  {outcome:>9}")

    print()
    total = sum(decision_counts.values())
    print("Sammanfattning:")
    for dec, n in decision_counts.items():
        pct = 100 * n / total
        print(f"  {dec:>9}: {n:>3} ({pct:.0f}%)")

    print()
    print("Tolkning:")
    print("  PASS      → modellen är trygg, svara autonomt")
    print("  EXPLORE   → flagga för ytterligare sampling/kontext")
    print("  ESCALATE  → för osäkert, lämna till människa")


if __name__ == "__main__":
    model = train_model()
    run_with_gates(model)
