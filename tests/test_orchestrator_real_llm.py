#!/usr/bin/env python3
"""
test_orchestrator_with_real_llm.py — End-to-end orchestration with real Groq API

Tests CognOS on three bedömning questions with actual LLM calls.
Demonstrates the upgrade from weak to strong synthesis.
"""

import sys
from pathlib import Path

# Add both cognos and parent paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cognos import CognOSOrchestrator

# Import LLM functions
try:
    from Jasper.jasper_brain import ask_groq
    GROQ_AVAILABLE = True
except ImportError:
    try:
        # Alternative: direct groq import
        from groq import Groq
        GROQ_AVAILABLE = True
        def ask_groq_direct(system: str, prompt: str) -> str:
            try:
                client = Groq()
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=800,
                )
                return response.choices[0].message.content.strip()
            except Exception:
                return None
        ask_groq = ask_groq_direct
    except ImportError:
        GROQ_AVAILABLE = False
        ask_groq = None


def real_llm_function(system: str, prompt: str) -> str:
    """
    Wrapper to use real Groq API.
    Falls back gracefully if API unavailable.
    """
    if not ask_groq:
        return None
    
    try:
        result = ask_groq(system, prompt)
        if result:
            return result
    except Exception as e:
        print(f"⚠ Groq API unavailable: {e}", file=sys.stderr)
        return None
    
    return None


def run_orchestration_test():
    """Run full orchestration on three bedömning questions."""
    
    print("\n" + "=" * 90)
    print("COGNOS ORCHESTRATOR — END-TO-END TEST WITH REAL LLM")
    print("=" * 90)
    print("\nSystem: Connecting to Groq API...\n")
    
    orchestrator = CognOSOrchestrator(
        llm_fn=real_llm_function,
        verbose=True,
        max_depth=3,
    )
    
    context = """CognOS — Epistemic Integrity Layer for AI Decision Systems.

Core Formula: C = p × (1 - Ue - Ua)

Three separable uncertainty types:
1. U_model (epistemic uncertainty): Internal model/sampling confidence variation
2. U_prompt (prompt uncertainty): How does answer change with different framings?
3. U_problem (problem uncertainty): Is the question itself well-posed?

HYPOTHESIS_V02 claims these can be measured independently and optimized.

This is the research question we're evaluating."""
    
    bedömning_questions = [
        {
            "title": "Falsifiability Assessment",
            "question": "How falsifiable is HYPOTHESIS_V02 in its current form?",
            "alternatives": [
                "A: Weakly falsifiable — theoretical criteria exist but operationalization unclear",
                "B: Partially falsifiable — some components testable, others require clarification",
                "C: Strongly falsifiable — clear disconfirmation conditions for all parts"
            ]
        },
        {
            "title": "Critical Weakness Identification",
            "question": "What is HYPOTHESIS_V02's most critical weakness right now?",
            "alternatives": [
                "A: Empirical foundation too narrow — only tested on AI confidence decisions",
                "B: U_prompt and U_model overlap too much — not sufficiently independent",
                "C: Integration strategy underspecified — how to act on uncertainty values"
            ]
        },
        {
            "title": "Validation Readiness",
            "question": "Should we run the full validation experiment now?",
            "alternatives": [
                "A: Yes — definitions are sufficient, start collecting data immediately",
                "B: Yes but — definitions strong enough, need 2 weeks prep for protocol",
                "C: No — need more groundwork on operational metrics and measurement"
            ]
        }
    ]
    
    results = []
    
    for i, q in enumerate(bedömning_questions, 1):
        print(f"\n{'#' * 90}")
        print(f"# BEDÖMNING {i}/3: {q['title']}")
        print(f"{'#' * 90}\n")
        
        try:
            result = orchestrator.orchestrate(
                question=q['question'],
                alternatives=q['alternatives'],
                context=context,
                n_samples=3,
            )
            
            results.append({
                'title': q['title'],
                'question': q['question'],
                'result': result,
            })
            
            # Display summary
            print(f"\n{'=' * 90}")
            print(f"RESULTAT: {q['title']}")
            print(f"{'=' * 90}")
            print(f"Beslut:              {result['decision'].upper()}")
            print(f"Confidence:          {result['confidence']:.3f} / 1.0")
            print(f"Iterationer:         {result['iterations']}")
            print(f"Konvergerat:         {'Ja ✓' if result['converged'] else 'Nej (mer djup möjlig)'}")
            
            if result.get('final_answer'):
                print(f"Svar:                {result['final_answer']}")
            
            # Show divergence analysis if present
            if result['layers'] and 'divergence' in result['layers'][-1]:
                div = result['layers'][-1]['divergence']
                print(f"\n🧠 DIVERGENSANALYS:")
                
                if isinstance(div, dict):
                    if div.get('majority_assumption'):
                        print(f"  Majoritet antar:  {div['majority_assumption'][:65]}")
                    if div.get('minority_assumption'):
                        print(f"  Minoritet antar:  {div['minority_assumption'][:65]}")
                    
                    if div.get('divergence_type'):
                        print(f"  Typ:              {div['divergence_type']}")
                    if div.get('integration_mode'):
                        print(f"  Integr. läge:     {div['integration_mode']}")
                    
                    if div.get('divergence_axes'):
                        print(f"  Geometri:         {len(div['divergence_axes'])} dimensioner")
                    
                    if div.get('meta_alternatives'):
                        print(f"  Nästa steg:       {len(div['meta_alternatives'])} alternativ")
                        for j, alt in enumerate(div['meta_alternatives'][:2], 1):
                            if isinstance(alt, str):
                                print(f"    {j}. {alt[:50]}")
                            elif isinstance(alt, dict) and 'label' in alt:
                                print(f"    {j}. {alt['label']}")
        
        except Exception as e:
            print(f"⚠ Error during orchestration: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'title': q['title'],
                'question': q['question'],
                'error': str(e),
            })
    
    # Final comparison
    print(f"\n\n{'=' * 90}")
    print("SAMMANFATTNING — ALLA BEDÖMNINGAR")
    print(f"{'=' * 90}\n")
    
    print("Bedömning              | Beslut    | Certainty | Iterationer | Status")
    print("-" * 90)
    
    for res in results:
        if 'error' in res:
            print(f"{res['title'][:25]:<25}| {'ERROR':<9} | {'—':<9} | {'—':<11} | Failed")
        else:
            r = res['result']
            status = "Konvergerat ✓" if r['converged'] else "Pågår"
            print(f"{res['title'][:25]:<25}| {r['decision']:<9} | {r['confidence']:>9.3f} | {r['iterations']:>11} | {status}")
    
    print(f"\n{'=' * 90}")
    print("✅ ORCHESTRATOR TEST SLUTFÖRT")
    print(f"{'=' * 90}\n")
    
    # Analysis
    successful = sum(1 for r in results if 'result' in r)
    with_divergence = sum(1 for r in results if 'result' in r and 
                         r['result']['layers'] and 
                         'divergence' in r['result']['layers'][-1])
    
    print(f"Statistik:")
    print(f"  • Lyckade bedömningar:         {successful}/3")
    print(f"  • Med divergensanalys:        {with_divergence}/3")
    print(f"  • Genomsnittlig säkerhet:     {sum(r['result']['confidence'] for r in results if 'result' in r) / successful if successful else 0:.3f}")
    print()
    
    return results


if __name__ == '__main__':
    results = run_orchestration_test()
