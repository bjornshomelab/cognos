#!/usr/bin/env python3
"""
CognOS — Uncertainty Gate

En gate som mäter epistemisk osäkerhet vid ett reasoning-steg och
returnerar beslut: pass / explore / escalate.

Designad för pipeline-placering vid varje reasoning-steg:

    hidden_0 → [Gate] → pass → hidden_1 → [Gate] → pass → ... → svar
                                                   ↓ escalate
                                                   stopp

Ue-källor:
    "entropy"  — softmax-entropi (kräver logits, validerad signal, positiv korrelation)
    "stream"   — inter-ström-divergens (kräver mHC-internals för rätt tecken)
    "variance" — varians i hidden state (enkel proxy)

OBS om "stream" utan mHC:
    I standard-lager (exp_007) är stream-divergens NEGATIVT korrelerad med Ue.
    Positiv korrelation kräver Birkhoff-routing (mHC). Sätt use_mhc_sign=True
    bara om hidden state kommer från ett tränat mHC-lager.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal, List, Optional

UeSource  = Literal["entropy", "stream", "variance"]
Decision  = Literal["pass", "explore", "escalate"]
CDecision = Literal["auto", "synthesize", "explore", "escalate"]


# ── Kärn-Ue-mätare ────────────────────────────────────────────────────────────

def _ue_entropy(vector: np.ndarray) -> float:
    """Softmax-entropi från logits, normaliserad till [0, 1]."""
    shifted = vector - vector.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()
    h = float(-(probs * np.log(probs + 1e-9)).sum())
    max_h = float(np.log(len(probs)))
    return h / max_h if max_h > 0 else 0.0


def _ue_stream(vector: np.ndarray, n_streams: int, mhc_sign: bool) -> float:
    """
    Inter-ström-divergens via statisk partitionering.
    Rätt tecken endast med mHC-routing (mhc_sign=True).
    """
    dim = len(vector)
    stream_dim = dim // n_streams
    if stream_dim == 0:
        return 0.0
    streams = vector[: n_streams * stream_dim].reshape(n_streams, stream_dim)
    mean = streams.mean(axis=0)
    div = float(np.linalg.norm(streams - mean, axis=1).mean())
    # Invertera om vi vet att vi INTE har mHC (exp_007-resultat)
    if not mhc_sign:
        # Normalisera till [0, 1] och invertera
        max_possible = float(np.linalg.norm(vector))
        if max_possible > 0:
            div = 1.0 - (div / max_possible)
    return div


def _ue_variance(vector: np.ndarray) -> float:
    """Varians i hidden state. Normaliserad proxy."""
    v = float(np.var(vector))
    # Normalisera grovt till rimlig skala
    return min(v / (float(np.mean(np.abs(vector))) ** 2 + 1e-9), 1.0)


# ── Gate ──────────────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    ue: float
    decision: Decision
    low: float
    high: float
    source: UeSource
    step: int = 0

    @property
    def passed(self) -> bool:
        return self.decision == "pass"

    def __repr__(self) -> str:
        icon = {"pass": "✓", "explore": "~", "escalate": "✗"}[self.decision]
        return (
            f"[Gate step={self.step}] {icon} {self.decision.upper()} "
            f"Ue={self.ue:.3f} (cutoffs: {self.low}/{self.high})"
        )


class UncertaintyGate:
    """
    Mäter Ue vid ett reasoning-steg och returnerar pass/explore/escalate.

    Args:
        low:        Ue under detta → pass (hög konfidens)
        high:       Ue över detta → escalate (för osäkert)
        source:     Ue-mätmetod: "entropy" | "stream" | "variance"
        n_streams:  Antal strömmar för source="stream"
        use_mhc_sign: True om hidden state från tränat mHC-lager
    """

    def __init__(
        self,
        low: float = 0.15,
        high: float = 0.45,
        source: UeSource = "entropy",
        n_streams: int = 4,
        use_mhc_sign: bool = False,
    ):
        if not (0 < low < high < 1):
            raise ValueError(f"Kräver 0 < low < high < 1, fick low={low} high={high}")
        self.low = low
        self.high = high
        self.source = source
        self.n_streams = n_streams
        self.use_mhc_sign = use_mhc_sign

    def measure(self, vector: np.ndarray) -> float:
        """Mät Ue från vektor (logits eller hidden state)."""
        v = np.asarray(vector, dtype=float).ravel()
        if self.source == "entropy":
            return _ue_entropy(v)
        elif self.source == "stream":
            return _ue_stream(v, self.n_streams, self.use_mhc_sign)
        else:  # variance
            return _ue_variance(v)

    def __call__(self, vector: np.ndarray, step: int = 0) -> GateResult:
        """Kör grinden. Returnerar GateResult med decision."""
        ue = self.measure(vector)
        if ue < self.low:
            decision: Decision = "pass"
        elif ue > self.high:
            decision = "escalate"
        else:
            decision = "explore"
        return GateResult(ue=ue, decision=decision, low=self.low,
                          high=self.high, source=self.source, step=step)


# ── CognOS Gate (med Ua-integration) ─────────────────────────────────────────

@dataclass
class CognOSGateResult:
    confidence: float
    decision: CDecision
    ue: float
    ua: float
    ua_source: str
    prediction: float
    step: int = 0

    def __repr__(self) -> str:
        icons = {"auto": "✓", "explore": "~", "escalate": "✗", "synthesize": "⟷"}
        icon = icons.get(self.decision, "?")
        return (
            f"[CognOSGate step={self.step}] {icon} {self.decision.upper()} "
            f"C={self.confidence:.3f} Ue={self.ue:.3f} Ua={self.ua:.3f} "
            f"(p={self.prediction:.3f}, Ua-källa: {self.ua_source})"
        )


class CognOSGate(UncertaintyGate):
    """
    Full CognOS-gate: Ue från hidden state/logits + semantisk Ua → C → beslut.

    Utökar UncertaintyGate med compute_confidence()-formeln:
        C = prediction × (1 − Ue − Ua)

    Beslutsvokabulär (matchar confidence.py):
        auto      — C ≥ threshold, agera autonomt
        explore   — C < threshold, Ue måttlig (sampla mer kontext)
        escalate  — Ue > high ELLER irreversibility hög (för riskfyllt)

    OBS: "synthesize" kräver multimodal-detektion via mc_predictions i
    compute_confidence(). Utan mc_predictions returneras aldrig "synthesize".
    Använd compute_confidence() direkt för full multimodal-hantering.

    Args:
        threshold:              C-tröskel för auto (default 0.72)
        irreversibility_override: Irr-nivå som alltid eskalerar (default 0.85)
        (övriga argument ärvs från UncertaintyGate)
    """

    def __init__(
        self,
        low: float = 0.15,
        high: float = 0.45,
        source: UeSource = "entropy",
        n_streams: int = 4,
        use_mhc_sign: bool = False,
        threshold: float = 0.72,
        irreversibility_override: float = 0.85,
    ):
        super().__init__(low=low, high=high, source=source,
                         n_streams=n_streams, use_mhc_sign=use_mhc_sign)
        self.threshold = threshold
        self.irr_override = irreversibility_override

    def __call__(  # type: ignore[override]
        self,
        vector: np.ndarray,
        ambiguity: Optional[float] = None,
        irreversibility: Optional[float] = None,
        blast_radius: Optional[float] = None,
        step: int = 0,
    ) -> CognOSGateResult:
        """
        Kör full CognOS-gate med valfria Ua-komponenter.

        Args:
            vector:          Logits eller hidden state (numpy array)
            ambiguity:       Instruktionstvetydighet [0, 1]
            irreversibility: Handlingens reversibilitet [0, 1]
            blast_radius:    Räckvidd av konsekvenser [0, 1]
            step:            Reasoning-steg (för logging)
        """
        v = np.asarray(vector, dtype=float).ravel()
        Ue = self.measure(v)

        # Extrahera top-1 prediction från logits via softmax
        shifted = v - v.max()
        probs = np.exp(shifted) / np.exp(shifted).sum()
        prediction = float(probs.max())

        # Ua — semantisk risk eller legacy-heuristik
        semantic = all(x is not None for x in [ambiguity, irreversibility, blast_radius])
        if semantic:
            Ua: float = float((ambiguity + irreversibility + blast_radius) / 3.0)
            ua_source = "semantic"
        else:
            Ua = float(2 * prediction * (1 - prediction))
            ua_source = "legacy"
        Ua = min(max(0.0, Ua), 1.0)

        # Beslutskonfidens
        C = float(max(0.0, min(1.0, prediction * (1 - Ue - Ua))))

        # Beslutslogik (prioritetsordning matchar compute_confidence)
        irr = irreversibility if irreversibility is not None else 0.0
        if irr >= self.irr_override and C < self.threshold:
            decision: CDecision = "escalate"
        elif C >= self.threshold:
            decision = "auto"
        elif Ue > self.high:
            decision = "escalate"
        else:
            decision = "explore"

        return CognOSGateResult(
            confidence=C, decision=decision,
            ue=Ue, ua=Ua, ua_source=ua_source,
            prediction=prediction, step=step,
        )


# ── Pipeline ──────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    outcome: Decision
    stopped_at: int        # index för sista körda steget
    history: List[GateResult] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = [f"Pipeline outcome: {self.outcome.upper()} (stopped_at={self.stopped_at})"]
        for r in self.history:
            lines.append(f"  {r}")
        return "\n".join(lines)


class ReasoningPipeline:
    """
    Kör en sekvens av UncertaintyGates över reasoning-steg.
    Stoppar vid första escalate.

    Exempel:
        gates = [UncertaintyGate(low=0.1, high=0.4)] * 3
        pipeline = ReasoningPipeline(gates)
        result = pipeline.run([h0, h1, h2])
    """

    def __init__(self, gates: List[UncertaintyGate]):
        self.gates = gates

    def run(self, hidden_states: List[np.ndarray]) -> PipelineResult:
        """
        Kör igenom alla gates.
        Om fler hidden_states än gates: kör bara min(len).
        Stoppar direkt vid escalate.
        """
        history: List[GateResult] = []
        n = min(len(self.gates), len(hidden_states))

        for i in range(n):
            result = self.gates[i](hidden_states[i], step=i)
            history.append(result)
            if result.decision == "escalate":
                return PipelineResult(outcome="escalate", stopped_at=i, history=history)

        # Sista steget avgör om vi utforskade eller passerade
        last = history[-1] if history else None
        final: Decision = "explore" if (last and last.decision == "explore") else "pass"
        return PipelineResult(outcome=final, stopped_at=n - 1, history=history)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CognOS Gate — Smoke test")
    print("=" * 52)

    gate = UncertaintyGate(low=0.15, high=0.45, source="entropy")

    # Tydlig klassifikation (låg Ue)
    confident_logits = np.array([8.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    print(gate(confident_logits, step=0))

    # Jämn distribution (hög Ue)
    uncertain_logits = np.ones(10)
    print(gate(uncertain_logits, step=1))

    # Måttlig osäkerhet
    medium_logits = np.array([2.0, 1.5, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    print(gate(medium_logits, step=2))

    print()
    print("Pipeline — 3-stegs reasoning:")
    pipeline = ReasoningPipeline([
        UncertaintyGate(low=0.15, high=0.45, source="entropy"),
        UncertaintyGate(low=0.15, high=0.45, source="entropy"),
        UncertaintyGate(low=0.15, high=0.45, source="entropy"),
    ])
    hidden_states = [confident_logits, medium_logits, uncertain_logits]
    result = pipeline.run(hidden_states)
    print(result)

    print()
    print("CognOSGate — med Ua-integration:")
    print("=" * 52)
    cgate = CognOSGate(low=0.15, high=0.45, threshold=0.72)

    # Låg risk, tydlig input → auto
    print(cgate(confident_logits,
                ambiguity=0.05, irreversibility=0.05, blast_radius=0.05, step=0))

    # Måttlig Ue + måttlig semantisk risk → explore
    print(cgate(medium_logits,
                ambiguity=0.30, irreversibility=0.20, blast_radius=0.20, step=1))

    # Hög irreversibilitet → escalate oavsett Ue
    print(cgate(medium_logits,
                ambiguity=0.10, irreversibility=0.90, blast_radius=0.30, step=2))

    # Hög Ue, ingen Ua-info → legacy-fallback
    print(cgate(uncertain_logits, step=3))
