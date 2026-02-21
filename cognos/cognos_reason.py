#!/usr/bin/env python3
"""
cognos_reason.py — The Recursive Epistemology Engine

This is the BRAIN of CognOS.

Not a confidence engine. A RECURSIVE EPISTEMOLOGY ENGINE.

def cognos_reason(question, context):
    layer1 = analyze(question)          # Voting + confidence
    layer2 = meta(layer1)                # Assumption extraction
    layer3 = meta_meta(layer2)           # Meta-iteration if divergence
    layer4 = synthesis(layer3)           # Integration strategy
    decision = convergence(layer4)       # Stop when stable
    return decision

This is what orchestrator.py implements under the hood.
But cognos_reason() is the CLEAN API WRAPPER.

Usage:
    from cognos_reason import cognos_reason
    
    result = cognos_reason(
        question="Is the hypothesis falsifiable?",
        alternatives=["Weakly", "Partially", "Strongly"],
        context="Hypothesis: C = p × (1 - Ue - Ua)"
    )
    
    print(result['decision'])           # 'auto' | 'explore' | 'synthesize' | 'escalate'
    print(result['final_answer'])        # The actual answer
    print(result['epistemic_state'])     # Full memory trace
"""

import sys
from pathlib import Path
from typing import Optional, Any

sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import CognOSOrchestrator
from epistemic_state import EpistemicState
from assumption_extraction import extract_all
from divergence_semantics import enhanced_frame_check


def cognos_reason(
    question: str,
    alternatives: list[str],
    context: str = "",
    n_samples: int = 5,
    max_depth: int = 4,
    convergence_threshold: float = 0.05,
    confidence_threshold: float = 0.72,
    multimodal_threshold: float = 0.20,
    llm_fn: Optional[Any] = None,
    verbose: bool = True,
    extract_assumptions: bool = True,
) -> dict:
    """
    The Recursive Epistemology Engine.
    
    This is the BRAIN. It orchestrates:
    - Layer 1: Voting + confidence calculation
    - Layer 2: Assumption extraction (if divergence)
    - Layer 3: Meta-iteration (recursive if needed)
    - Layer 4: Synthesis + integration strategy
    - Layer 5: Convergence detection (stop when stable)
    
    Input:
      question: The question to reason about
      alternatives: List of possible answers
      context: Background context for reasoning
      n_samples: How many LLM samples per voting round
      max_depth: Maximum meta-recursion depth
      convergence_threshold: When to stop recursion (confidence change < threshold)
      confidence_threshold: C threshold for AUTO vs EXPLORE
      multimodal_threshold: Ue separation threshold for detecting bimodality
      llm_fn: Optional custom LLM function
      verbose: Print reasoning trace
      extract_assumptions: Run full assumption extraction engine
    
    Returns: {
        'decision': str,                    # 'auto' | 'explore' | 'synthesize' | 'escalate'
        'confidence': float,                # Final confidence score
        'final_answer': str,                # The actual answer
        'epistemic_state': EpistemicState,  # Full memory trace
        'layers': list[dict],               # Each reasoning layer
        'converged': bool,                  # Did reasoning converge?
        'iterations': int,                  # Number of meta-iterations
        'assumption_analysis': dict,        # Latent assumptions, perspective mapping, ontology
        'frame_check': dict,                # Is question well-posed?
    }
    """
    
    # Create epistemic state memory
    epistemic_state = EpistemicState()
    
    # Layer 0: Frame validation (check if question is well-posed)
    if verbose:
        print("\n" + "="*80)
        print("🧠 COGNOS REASONING ENGINE")
        print("="*80)
        print(f"\n📝 Question: {question}")
        print(f"🎯 Alternatives: {len(alternatives)}")
        print(f"📚 Context: {len(context)} chars")
        print(f"🔧 Config: n={n_samples}, depth={max_depth}, C_thresh={confidence_threshold:.2f}")
    
    frame_check = enhanced_frame_check(question, llm_fn)
    
    if verbose:
        print(f"\n🔍 Frame Check: {'✅ Well-framed' if frame_check['is_well_framed'] else '⚠️ ' + frame_check['problem_type']}")
        if not frame_check['is_well_framed']:
            print(f"   Issues: {frame_check.get('specific_issues', [])}")
            print(f"   Missing: {frame_check.get('missing_specifications', [])}")
            if frame_check.get('reframed_question'):
                print(f"   Suggested: {frame_check['reframed_question']}")
    
    # Record frame check in epistemic state
    epistemic_state.record_frame(
        iteration=0,
        original_question=question,
        is_well_framed=frame_check['is_well_framed'],
        problem_type=frame_check.get('problem_type'),
        reframed_question=frame_check.get('reframed_question'),
        frame_quality_score=1.0 if frame_check['is_well_framed'] else 0.5,
    )
    
    # Use reframed question if necessary
    if not frame_check['is_well_framed'] and frame_check.get('reframed_question'):
        question = frame_check['reframed_question']
    
    # Layer 1-4: Run orchestrator (voting → confidence → divergence → convergence → recursion)
    orchestrator = CognOSOrchestrator(
        llm_fn=llm_fn,
        max_depth=max_depth,
        convergence_threshold=convergence_threshold,
        confidence_threshold=confidence_threshold,
        multimodal_threshold=multimodal_threshold,
        verbose=verbose,
    )
    
    orchestration_result = orchestrator.orchestrate(
        question=question,
        alternatives=alternatives,
        context=context,
        n_samples=n_samples,
    )
    
    # Populate epistemic state from orchestration layers
    for layer in orchestration_result['layers']:
        iteration = layer['iteration']
        
        # Record confidence
        epistemic_state.record_confidence(
            iteration=iteration,
            confidence=layer['confidence'],
            epistemic_ue=layer['epistemic_ue'],
            aleatoric_ua=layer['aleatoric_ua'],
            decision=layer['decision'],
            epistemic_gain=layer.get('epistemic_gain'),  # From strong_synthesis if available
        )
        
        # Record assumptions (if divergence was analyzed)
        if 'divergence' in layer:
            div = layer['divergence']
            epistemic_state.record_assumptions(
                iteration=iteration,
                majority_assumption=div.get('majority_assumption', ''),
                minority_assumption=div.get('minority_assumption', ''),
                divergence_type=div.get('divergence_type', 'unknown'),
                divergence_source=div.get('divergence_source', ''),
                geometry=div.get('divergence_axes', []),
                integration_mode=div.get('integration_mode', 'unknown'),
            )
        
        # Record perspectives
        if layer.get('votes'):
            vote_dist = layer['votes']
            majority = layer.get('majority_choice', '')
            alternatives_list = layer.get('alternatives', alternatives)
            
            epistemic_state.record_perspectives(
                iteration=iteration,
                perspectives=alternatives_list,
                vote_distribution=vote_dist,
                is_multimodal=layer.get('is_multimodal', False),
                dominant_perspective=majority,
                alternative_perspectives=[k for k in vote_dist.keys() if k != majority],
            )
    
    # Optional: Run assumption extraction engine (latent assumptions, perspective mapping, ontology)
    assumption_analysis = {}
    if extract_assumptions and llm_fn:
        if verbose:
            print("\n🔬 Running Assumption Extraction Engine...")
        
        assumption_analysis = extract_all(
            question=question,
            alternatives=alternatives,
            context=context,
            llm_fn=llm_fn,
        )
        
        if verbose:
            print(f"   Latent assumptions: {len(assumption_analysis['latent_assumptions'])}")
            print(f"   Perspective mappings: {len(assumption_analysis['perspective_mappings'])}")
            print(f"   Ontology tags: {len(assumption_analysis['ontology_tags'])}")
    
    # Final result
    if verbose:
        print("\n" + "="*80)
        print("📊 EPISTEMIC STATE SUMMARY")
        print("="*80)
        summary = epistemic_state.summary()
        for key, val in summary.items():
            print(f"   {key}: {val}")
        print("="*80 + "\n")
    
    return {
        'decision': orchestration_result['decision'],
        'confidence': orchestration_result['confidence'],
        'final_answer': orchestration_result.get('final_answer'),
        'epistemic_state': epistemic_state,
        'layers': orchestration_result['layers'],
        'converged': orchestration_result.get('converged', False),
        'iterations': orchestration_result.get('iterations', 0),
        'assumption_analysis': assumption_analysis,
        'frame_check': frame_check,
    }


def analyze(question: str, alternatives: list[str], context: str = "", llm_fn: Optional[Any] = None) -> dict:
    """Layer 1: Voting + confidence calculation."""
    return cognos_reason(question, alternatives, context, max_depth=1, llm_fn=llm_fn, extract_assumptions=False, verbose=False)


def meta(result: dict) -> dict:
    """Layer 2: Assumption extraction from divergence."""
    # Already done in cognos_reason, so just return
    return result


def meta_meta(result: dict) -> dict:
    """Layer 3: Meta-iteration if divergence persists."""
    # Already done in cognos_reason recursion
    return result


def synthesis(result: dict) -> dict:
    """Layer 4: Integration strategy."""
    # Already done in cognos_reason
    return result


def convergence(result: dict) -> str:
    """Layer 5: Return decision when stable."""
    return result['decision']


if __name__ == '__main__':
    """Demo: Run cognos_reason on a bedömning question."""
    
    result = cognos_reason(
        question="How falsifiable is HYPOTHESIS_V02 in its current form?",
        alternatives=[
            "Weakly falsifiable — lacks clear thresholds",
            "Partially falsifiable but requires stricter criteria",
            "Strongly falsifiable with clear operational definitions"
        ],
        context="""CognOS is an epistemic integrity layer for AI systems.
        
Formula: C = p × (1 - Ue - Ua)

Three uncertainty types:
- U_model: internal epistemic uncertainty
- U_prompt: format-induced uncertainty  
- U_problem: ill-posedness of the question itself

The HYPOTHESIS_V02 proposes these are separable and testable.""",
        n_samples=3,
        verbose=True,
    )
    
    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)
    print(f"Decision: {result['decision'].upper()}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Answer: {result['final_answer']}")
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print("="*80)
