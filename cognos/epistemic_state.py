#!/usr/bin/env python3
"""
epistemic_state.py — Memory structure for recursive epistemology.

This enables true meta-reasoning by tracking:
- assumptions_history: What assumptions were made at each layer
- confidence_history: How confidence evolved across iterations
- frame_history: How the question was transformed
- perspective_history: Which perspectives emerged and converged

Without this, recursion is blind. With it, meta-recursion can see its own process.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameCheckpoint:
    """A snapshot of how the question was framed at a specific layer."""
    iteration: int
    original_question: str
    is_well_framed: bool
    problem_type: Optional[str] = None
    reframed_question: Optional[str] = None
    frame_quality_score: float = 0.0


@dataclass
class AssumptionCheckpoint:
    """A snapshot of assumptions extracted at a specific layer."""
    iteration: int
    majority_assumption: str
    minority_assumption: str
    divergence_type: str  # epistemic | normative | scope | cost_of_error
    divergence_source: str
    geometry: list[dict] = field(default_factory=list)  # divergence_axes
    integration_mode: str = "unknown"  # reframe | tradeoff | empirical_test | clarification


@dataclass
class PerspectiveCheckpoint:
    """A snapshot of which perspectives were considered."""
    iteration: int
    perspectives: list[str]
    vote_distribution: dict[str, int]
    is_multimodal: bool
    dominant_perspective: str
    alternative_perspectives: list[str]


@dataclass
class ConfidenceCheckpoint:
    """A snapshot of confidence and uncertainty at a specific layer."""
    iteration: int
    confidence: float
    epistemic_ue: float
    aleatoric_ua: float
    decision: str  # auto | explore | synthesize | escalate
    epistemic_gain: Optional[float] = None  # Only after synthesis


class EpistemicState:
    """
    Memory structure for recursive epistemology.
    
    Tracks:
    - Frame evolution (how question was transformed)
    - Assumption evolution (what was believed at each layer)
    - Confidence evolution (how certainty changed)
    - Perspective evolution (which views emerged/converged)
    
    Enables meta-recursion to:
    - Detect when assumptions stabilize (convergence)
    - Identify when reframing improved confidence
    - See when perspectives collapsed to consensus
    - Measure epistemic gain from synthesis
    """
    
    def __init__(self):
        self.frames: list[FrameCheckpoint] = []
        self.assumptions: list[AssumptionCheckpoint] = []
        self.perspectives: list[PerspectiveCheckpoint] = []
        self.confidences: list[ConfidenceCheckpoint] = []
        
        # Quick-access histories (for backward compatibility)
        self.confidence_history: list[float] = []
        self.assumption_history: list[str] = []
        self.frame_history: list[str] = []
    
    def record_frame(
        self,
        iteration: int,
        original_question: str,
        is_well_framed: bool,
        problem_type: Optional[str] = None,
        reframed_question: Optional[str] = None,
        frame_quality_score: float = 0.0,
    ):
        """Record a frame checkpoint."""
        checkpoint = FrameCheckpoint(
            iteration=iteration,
            original_question=original_question,
            is_well_framed=is_well_framed,
            problem_type=problem_type,
            reframed_question=reframed_question,
            frame_quality_score=frame_quality_score,
        )
        self.frames.append(checkpoint)
        self.frame_history.append(reframed_question or original_question)
    
    def record_assumptions(
        self,
        iteration: int,
        majority_assumption: str,
        minority_assumption: str,
        divergence_type: str,
        divergence_source: str,
        geometry: Optional[list[dict]] = None,
        integration_mode: str = "unknown",
    ):
        """Record an assumption checkpoint."""
        checkpoint = AssumptionCheckpoint(
            iteration=iteration,
            majority_assumption=majority_assumption,
            minority_assumption=minority_assumption,
            divergence_type=divergence_type,
            divergence_source=divergence_source,
            geometry=geometry or [],
            integration_mode=integration_mode,
        )
        self.assumptions.append(checkpoint)
        self.assumption_history.append(divergence_source)
    
    def record_perspectives(
        self,
        iteration: int,
        perspectives: list[str],
        vote_distribution: dict[str, int],
        is_multimodal: bool,
        dominant_perspective: str,
        alternative_perspectives: list[str],
    ):
        """Record a perspective checkpoint."""
        checkpoint = PerspectiveCheckpoint(
            iteration=iteration,
            perspectives=perspectives,
            vote_distribution=vote_distribution,
            is_multimodal=is_multimodal,
            dominant_perspective=dominant_perspective,
            alternative_perspectives=alternative_perspectives,
        )
        self.perspectives.append(checkpoint)
    
    def record_confidence(
        self,
        iteration: int,
        confidence: float,
        epistemic_ue: float,
        aleatoric_ua: float,
        decision: str,
        epistemic_gain: Optional[float] = None,
    ):
        """Record a confidence checkpoint."""
        checkpoint = ConfidenceCheckpoint(
            iteration=iteration,
            confidence=confidence,
            epistemic_ue=epistemic_ue,
            aleatoric_ua=aleatoric_ua,
            decision=decision,
            epistemic_gain=epistemic_gain,
        )
        self.confidences.append(checkpoint)
        self.confidence_history.append(confidence)
    
    def get_assumption_stability(self) -> float:
        """
        Measure how stable assumptions are across iterations.
        
        Returns: stability score (0.0 = chaotic, 1.0 = fully stable)
        """
        if len(self.assumptions) < 2:
            return 0.0
        
        # Compare consecutive divergence sources
        consecutive_same = 0
        for i in range(1, len(self.assumptions)):
            prev = self.assumptions[i-1].divergence_source
            curr = self.assumptions[i].divergence_source
            # Simple text similarity (could be improved with embeddings)
            if prev.lower() == curr.lower():
                consecutive_same += 1
        
        stability = consecutive_same / (len(self.assumptions) - 1)
        return stability
    
    def get_confidence_trend(self) -> str:
        """
        Determine if confidence is increasing, decreasing, or stable.
        
        Returns: "increasing" | "decreasing" | "stable" | "chaotic"
        """
        if len(self.confidence_history) < 2:
            return "stable"
        
        recent = self.confidence_history[-3:]  # Last 3 iterations
        if len(recent) < 2:
            return "stable"
        
        diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        avg_diff = sum(diffs) / len(diffs)
        
        if abs(avg_diff) < 0.05:
            return "stable"
        elif avg_diff > 0.05:
            return "increasing"
        elif avg_diff < -0.05:
            return "decreasing"
        else:
            # Mixed directions = chaotic
            if any(d > 0 for d in diffs) and any(d < 0 for d in diffs):
                return "chaotic"
            return "stable"
    
    def get_total_epistemic_gain(self) -> float:
        """
        Sum epistemic gain across all synthesis steps.
        
        Returns: total epistemic gain (higher = more understanding achieved)
        """
        gains = [c.epistemic_gain for c in self.confidences if c.epistemic_gain is not None]
        return sum(gains) if gains else 0.0
    
    def is_converged(self, threshold: float = 0.05) -> tuple[bool, str]:
        """
        Check if the epistemic state has converged.
        
        Convergence = stable assumptions + stable confidence + no new perspectives
        
        Returns: (converged: bool, reason: str)
        """
        if len(self.confidences) < 2:
            return False, "Insufficient iterations"
        
        # Check confidence stability
        conf_trend = self.get_confidence_trend()
        if conf_trend == "chaotic":
            return False, "Confidence is chaotic"
        
        # Check assumption stability
        assumption_stability = self.get_assumption_stability()
        if assumption_stability < 0.7:
            return False, f"Assumptions unstable (stability={assumption_stability:.2f})"
        
        # Check if last 2 iterations had same decision type
        recent_decisions = [c.decision for c in self.confidences[-2:]]
        if len(set(recent_decisions)) > 1:
            return False, "Decision type changed in recent iterations"
        
        # Check if confidence change is small
        if len(self.confidence_history) >= 2:
            recent_change = abs(self.confidence_history[-1] - self.confidence_history[-2])
            if recent_change > threshold:
                return False, f"Confidence still changing (Δ={recent_change:.3f})"
        
        return True, "Assumptions stable, confidence stable, decision consistent"
    
    def summary(self) -> dict:
        """Return a summary dictionary of the epistemic state."""
        return {
            'iterations': len(self.confidences),
            'frames_checked': len(self.frames),
            'assumptions_extracted': len(self.assumptions),
            'perspectives_explored': len(self.perspectives),
            'confidence_trend': self.get_confidence_trend(),
            'assumption_stability': self.get_assumption_stability(),
            'total_epistemic_gain': self.get_total_epistemic_gain(),
            'converged': self.is_converged()[0],
            'convergence_reason': self.is_converged()[1],
        }
