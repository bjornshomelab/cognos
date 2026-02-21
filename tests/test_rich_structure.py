#!/usr/bin/env python3
"""
test_rich_divergence_structure.py — Demonstrate upgraded synthesize_reason()

Shows:
- divergence_type classification
- integration_mode (reframe | tradeoff | empirical_test | clarification)
- divergence_axes (geometric mapping)
- meta_alternatives (dynamic problem-solving options)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cognos.divergence_semantics import synthesize_reason


def mock_llm_fn(system: str, prompt: str) -> str:
    """Mock LLM that returns rich structured response."""
    return """{
        "majority_assumption": "Falsifiability requires clear theoretical disconfirmation conditions",
        "minority_assumption": "Falsifiability requires practical operational measurement protocols",
        "divergence_source": "Uncertainty about what counts as falsifiable (theory vs practice)",
        "divergence_type": "epistemic",
        "divergence_axes": [
            {
                "dimension": "Epistemological certainty",
                "majority_position": 0.85,
                "minority_position": 0.25,
                "interpretation": "Majority: Theory sufficient. Minority: Need practical measures."
            },
            {
                "dimension": "Implementation complexity",
                "majority_position": 0.2,
                "minority_position": 0.8,
                "interpretation": "Majority: Simple criteria. Minority: Complex protocol needed."
            }
        ],
        "integration_strategy": "Define falsifiability at two levels: theoretical + operational",
        "integration_mode": "clarification",
        "meta_question": "What level of operational detail do we actually need for testing?",
        "meta_alternatives": [
            "Prioritize theoretical rigor (philosophy-first approach)",
            "Prioritize empirical operationalization (engineering-first approach)",
            "Create dual-track both theory + operational specs simultaneously"
        ],
        "is_resolvable": true
    }"""


def test_rich_structure():
    """Test the upgraded synthesize_reason with all new fields."""
    print("=" * 90)
    print("TEST: Rich Divergence Structure Upgrade")
    print("=" * 90)
    
    result = synthesize_reason(
        question="How falsifiable is HYPOTHESIS_V02?",
        alternatives=[
            "A: Weakly falsifiable",
            "B: Partially falsifiable",
            "C: Strongly falsifiable"
        ],
        vote_distribution={"B": 3, "C": 2},
        confidence=0.65,
        is_multimodal=True,
        context="Testing new structure",
        llm_fn=mock_llm_fn,
    )
    
    print("\n✅ CORE ASSUMPTIONS (clear before):")
    print(f"  Majority: {result['majority_assumption']}")
    print(f"  Minority: {result['minority_assumption']}")
    
    print(f"\n✅ DIVERGENCE CLASSIFICATION (NEW):")
    print(f"  Type: {result['divergence_type']}")
    print(f"  Source: {result['divergence_source']}")
    
    print(f"\n✅ CONFLICT GEOMETRY (NEW: actionable dimensions):")
    if result['divergence_axes']:
        for axis in result['divergence_axes']:
            print(f"\n  Dimension: {axis['dimension']}")
            print(f"    Majority position: {axis['majority_position']:+.1f}")
            print(f"    Minority position: {axis['minority_position']:+.1f}")
            print(f"    Interpretation: {axis['interpretation']}")
    else:
        print("  (No axes — LLM fallback)")
    
    print(f"\n✅ INTEGRATION STRATEGY (NEW: operationally concrete):")
    print(f"  Mode: {result['integration_mode']}")
    print(f"  Strategy: {result['integration_strategy']}")
    
    print(f"\n✅ META-ALTERNATIVES (NEW: dynamic next steps)")
    if result['meta_alternatives']:
        print(f"  Meta-question: {result['meta_question']}")
        print(f"  Options ({len(result['meta_alternatives'])} alternatives):")
        for i, alt in enumerate(result['meta_alternatives'], 1):
            print(f"    {i}. {alt}")
    
    print(f"\n✅ RESOLVABILITY: {result['is_resolvable']}")
    
    print("\n" + "=" * 90)
    print("ASSESSMENT")
    print("=" * 90)
    
    checks = {
        "Has divergence_type": result.get('divergence_type') is not None,
        "Has divergence_axes": len(result.get('divergence_axes', [])) > 0,
        "Has integration_mode": result.get('integration_mode') is not None,
        "Has meta_alternatives": len(result.get('meta_alternatives', [])) > 0,
        "Integration is actionable": len(result['integration_strategy']) > 20,
    }
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    passed = sum(checks.values())
    total = len(checks)
    print(f"\nScore: {passed}/{total} new structure features present")
    
    if passed == total:
        print("\n🎯 UPGRADE SUCCESSFUL: synthesize_reason() now returns rich, operationally structured output")
    else:
        print(f"\n⚠ Partial: {total - passed} features missing (check LLM response)")
    
    print("=" * 90 + "\n")


if __name__ == '__main__':
    test_rich_structure()
