"""
CognOS — Epistemic Integrity Layer for Agentic AI

The operating system for decision-aware AI systems.

Main API:
    from cognos import cognos_reason
    
    result = cognos_reason(
        question="Is this falsifiable?",
        alternatives=["Weakly", "Partially", "Strongly"],
        context="Background information"
    )

This is a RECURSIVE EPISTEMOLOGY ENGINE, not just a confidence calculator.
"""

__version__ = "0.3.0"
__author__ = "Björn Wikström"
__license__ = "MIT"

# Core reasoning API
from .cognos_reason import (
    cognos_reason,
    analyze,
    meta,
    meta_meta,
    synthesis,
    convergence,
)

# Orchestration
from .orchestrator import CognOSOrchestrator

# Epistemic state memory
from .epistemic_state import (
    EpistemicState,
    FrameCheckpoint,
    AssumptionCheckpoint,
    PerspectiveCheckpoint,
    ConfidenceCheckpoint,
)

# Layer 1: Confidence
from .confidence import compute_confidence

# Layer 2: Divergence semantics
from .divergence_semantics import (
    synthesize_reason,
    frame_transform,
    enhanced_frame_check,
    convergence_check,
)

# Assumption extraction engine
from .assumption_extraction import (
    extract_all,
    extract_latent_assumptions,
    map_perspectives,
    tag_ontology,
    AssumptionType,
    PerspectiveFrame,
    LatentAssumption,
    PerspectiveMapping,
    OntologyTag,
)

# Strong synthesis
try:
    from .strong_synthesis import (
        synthesize_strong,
        extract_assumptions_and_geometry,
        generate_integration_strategy,
        generate_meta_alternatives,
        compute_epistemic_gain,
    )
    STRONG_SYNTHESIS_AVAILABLE = True
except ImportError:
    STRONG_SYNTHESIS_AVAILABLE = False

__all__ = [
    # Main API
    "cognos_reason",
    "analyze",
    "meta",
    "meta_meta",
    "synthesis",
    "convergence",
    
    # Orchestration
    "CognOSOrchestrator",
    
    # Epistemic state
    "EpistemicState",
    "FrameCheckpoint",
    "AssumptionCheckpoint",
    "PerspectiveCheckpoint",
    "ConfidenceCheckpoint",
    
    # Layer 1
    "compute_confidence",
    
    # Layer 2
    "synthesize_reason",
    "frame_transform",
    "enhanced_frame_check",
    "convergence_check",
    
    # Assumption extraction
    "extract_all",
    "extract_latent_assumptions",
    "map_perspectives",
    "tag_ontology",
    "AssumptionType",
    "PerspectiveFrame",
    "LatentAssumption",
    "PerspectiveMapping",
    "OntologyTag",
    
    # Strong synthesis
    "synthesize_strong",
    "extract_assumptions_and_geometry",
    "generate_integration_strategy",
    "generate_meta_alternatives",
    "compute_epistemic_gain",
    "STRONG_SYNTHESIS_AVAILABLE",
]

