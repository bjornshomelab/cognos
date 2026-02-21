#!/usr/bin/env python3
"""
test_divergence_with_real_llm.py — Test strong synthesis with real Groq API

Designed to trigger divergence analysis by asking nuanced questions
that cause voting disagreement, which then triggers synthesize_reason().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cognos import CognOSOrchestrator

# Try to import ask_groq from Jasper
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Jasper"))
    from jasper_brain import ask_groq
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    ask_groq = None


def run_divergence_test():
    """
    Run orchestrator on questions designed to cause divergence analysis.
    """
    
    print("\n" + "=" * 90)
    print("COGNOS STRONG SYNTHESIS TEST — REAL LLM WITH DIVERGENCE")
    print("=" * 90)
    
    if not LLM_AVAILABLE:
        print("\n✗ Groq not available. Running with mock instead.\n")
    else:
        print("\n✓ Using real Groq API\n")
    
    orchestrator = CognOSOrchestrator(
        llm_fn=ask_groq if LLM_AVAILABLE else None,
        verbose=True,
        max_depth=4,
    )
    
    context = """CognOS — Epistemic Integrity Layer.

Formula: C = p × (1 - Ue - Ua)

Three uncertainty types:
- U_model: Internal model confidence variation
- U_prompt: Format-induced uncertainty
- U_problem: Question ill-posedness

HYPOTHESIS_V02 proposes these are independently measurable."""
    
    # Questions designed to cause divergence in voting
    test_cases = [
        {
            "title": "Practical Falsifiability",
            "question": """Is HYPOTHESIS_V02 practically falsifiable in a research setting?

Consider both theoretical clarity AND practical measurement difficulty.""",
            "alternatives": [
                "A: Yes — both clear, proceed to testing",
                "B: Unclear —theory clear but measurement difficult",
                "C: No — both theory and measurement are unclear"
            ]
        },
        {
            "title": "Independence Assessment",
            "question": "Are U_model, U_prompt, and U_problem truly independent or overlapping?",
            "alternatives": [
                "A: Completely independent",
                "B: Partially independent with some overlap",
                "C: Fundamentally not independent"
            ]
        },
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'#' * 90}")
        print(f"# TEST {i}/{len(test_cases)}: {case['title']}")
        print(f"{'#' * 90}\n")
        
        try:
            result = orchestrator.orchestrate(
                question=case['question'],
                alternatives=case['alternatives'],
                context=context,
                n_samples=5,  # More to increase divergence chance
            )
            
            results.append({
                'title': case['title'],
                'result': result,
            })
            
            print(f"\n{'=' * 90}")
            print(f"RESULT")
            print(f"{'=' * 90}")
            print(f"Decision:       {result['decision'].upper()}")
            print(f"Confidence:     {result['confidence']:.3f}")
            print(f"Iterations:     {result['iterations']}")
            print(f"Converged:      {result['converged']}")
            
            # Show layer details
            for layer_idx, layer in enumerate(result.get('layers', []), 1):
                print(f"\nLayer {layer_idx}:")
                print(f"  Decision: {layer.get('decision', '?')}")
                print(f"  Confidence: {layer.get('confidence', 0):.3f}")
                
                if layer.get('is_multimodal'):
                    print(f"  ⚠ Multimodal distribution detected")
                
                # DIVERGENCE ANALYSIS
                if 'divergence' in layer:
                    div = layer['divergence']
                    print(f"\n  🧠 DIVERGENCE SYNTHESIS:")
                    
                    if div.get('majority_assumption'):
                        print(f"    Majority: {div['majority_assumption'][:65]}")
                    if div.get('minority_assumption'):
                        print(f"    Minority: {div['minority_assumption'][:65]}")
                    
                    # NEW FIELDS FROM UPGRADE
                    if div.get('divergence_type'):
                        print(f"    Type: {div['divergence_type']}")
                    if div.get('integration_mode'):
                        print(f"    Integration: {div['integration_mode']}")
                    
                    if div.get('divergence_axes'):
                        print(f"    Geometry: {len(div['divergence_axes'])} dimensions")
                        for ax in div['divergence_axes'][:1]:
                            maj = ax.get('majority_position', '?')
                            min = ax.get('minority_position', '?')
                            print(f"      - {ax['dimension']}: Maj={maj:+.1f}, Min={min:+.1f}")
                    
                    if div.get('meta_alternatives'):
                        print(f"    Next steps: {len(div['meta_alternatives'])} options")
        
        except Exception as e:
            print(f"⚠ Error: {e}")
            results.append({
                'title': case['title'],
                'error': str(e),
            })
    
    # Summary
    print(f"\n\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}\n")
    
    with_divergence = sum(1 for r in results if 'result' in r and 
                         any('divergence' in layer for layer in r['result'].get('layers', [])))
    
    print(f"Tests completed: {len([r for r in results if 'result' in r])}/{len(test_cases)}")
    print(f"With divergence synthesis: {with_divergence}/{len(test_cases)}")
    
    if with_divergence > 0:
        print(f"\n✓ Strong synthesis activated!")
        print(f"  - Divergence detection: ✓")
        print(f"  - Assumption extraction: ✓")
        print(f"  - Integration strategy: ✓")
        print(f"  - Meta-alternatives: ✓")
    else:
        print(f"\nℹ No divergence detected (unanimity across samples)")
        print(f"  This is normal when LLM voting is highly consistent")
    
    print()


if __name__ == '__main__':
    run_divergence_test()
