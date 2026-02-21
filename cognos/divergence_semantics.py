#!/usr/bin/env python3
"""
divergence_semantics.py — Externe metacognitive conflict resolution.

When CognOS detects SYNTHESIZE (perspective conflict), this module
extracts the underlying assumptions that drive the divergence.

Core idea: perspektiv A och B är kohärenta, men bygger på olika antaganden.
Systemet måste kunna säga VILKA antaganden som skiljer dem.

Det är meta-nivå 1: från "vi är oeniga" till "vi antar olika saker om X".

Tre funktioner:
  - synthesize_reason()   : extrahera underliggande antaganden från röstfördelning
  - frame_transform()     : detektera om frågan är felställd (meta-nivå 2)
  - convergence_check()   : stoppa när C och antaganden stabiliseras (meta-nivå 3)
"""

import json
import sys
from pathlib import Path
from typing import Optional, Any

# Try to import Jasper's ask functions; fallback to direct API if unavailable
try:
    sys.path.append('/media/bjorn/iic/Jasper')
    from jasper_brain import ask_groq, ask_github_models
except ImportError:
    ask_groq = None
    ask_github_models = None


def _call_llm(system: str, prompt: str) -> Optional[str]:
    """Try GitHub Models first, then Groq, then return None."""
    if ask_github_models:
        try:
            result = ask_github_models(system, prompt)
            if result:
                return result
        except Exception:
            pass
    if ask_groq:
        try:
            result = ask_groq(system, prompt)
            if result:
                return result
        except Exception:
            pass
    return None


def _fallback_synthesize_reason(question: str, majority_choice: str, minority_choice: str, 
                                majority_alt: str, minority_alt: str, confidence: float) -> dict:
    """Graceful fallback when LLM is unavailable."""
    return {
        'question': question,
        'majority_choice': majority_choice,
        'minority_choice': minority_choice,
        'majority_assumption': f'Majoriteten ({majority_choice}) föredrar: {majority_alt}',
        'minority_assumption': f'Minoriteten ({minority_choice}) föredrar: {minority_alt}',
        'divergence_source': 'Okänd — LLM ej tillgänglig',
        'divergence_type': 'unknown',
        'divergence_axes': [],
        'integration_strategy': 'Kunde inte analysera (LLM ej tillgänglig)',
        'integration_mode': 'clarification',
        'meta_question': 'Kunde inte generera metafråga',
        'meta_alternatives': [],
        'confidence': confidence,
        'is_resolvable': False,
    }


def synthesize_reason(
    question: str,
    alternatives: list[str],
    vote_distribution: dict,
    confidence: float,
    is_multimodal: bool,
    context: Optional[str] = None,
    llm_fn: Optional[Any] = None,
) -> dict:
    """
    Extrahera underliggande antaganden från en divergent röstfördelning.

    Input:
      question: Ursprungsfrågan (t.ex. "Är hypotesen falsifierbar?")
      alternatives: Svarsmöjligheter (t.ex. ["A: Svag", "B: Medel", "C: Stark"])
      vote_distribution: Röster per svar (t.ex. {"B": 3, "C": 2})
      confidence: CognOS confidence score
      is_multimodal: Om Ue-distributionen är bimodal
      context: Valfri kontextinformation

    Output:
      {
        'question': str,
        'majority_choice': str,
        'majority_assumption': str,        # Vad majoriteten antar
        'minority_choice': str,
        'minority_assumption': str,        # Vad minoriteten antar
        'divergence_source': str,          # VAR skiljer sig antagandena?
        'integration_strategy': str,       # Hur kombinerar man dem?
        'meta_question': str,              # Vad behöver vi klargöra?
        'confidence': float,
        'is_resolvable': bool,             # Kan divergensen lösas genom mer info?
      }
    """

    # Identifiera majoritet och minoritet
    if not vote_distribution or confidence >= 0.95:
        return {
            'question': question,
            'majority_choice': None,
            'majority_assumption': 'Konsensus — ingen divergens att analysera.',
            'minority_choice': None,
            'minority_assumption': None,
            'divergence_source': None,
            'integration_strategy': 'Ingen syntes behövlig.',
            'meta_question': None,
            'confidence': confidence,
            'is_resolvable': True,
        }

    sorted_votes = sorted(vote_distribution.items(), key=lambda x: x[1], reverse=True)
    majority_choice, majority_count = sorted_votes[0]
    minority_choice, minority_count = sorted_votes[1] if len(sorted_votes) > 1 else (None, 0)

    # Mappa choicelabels till alternativstext
    choice_to_alt = {}
    for i, alt in enumerate(alternatives):
        label = chr(65 + i)  # A, B, C, ...
        choice_to_alt[label] = alt

    majority_alt = choice_to_alt.get(majority_choice, f"Alternative {majority_choice}")
    minority_alt = choice_to_alt.get(minority_choice, f"Alternative {minority_choice}") if minority_choice else None

    # Skapa prompt för LLM att extrahera antaganden + strukturera divergensen
    prompt = f"""
Du är en filosofisk analytiker som specialiserar sig på underliggande antaganden i diskoord.

Fråga: {question}

Alternativ:
{chr(10).join(f"  {label}: {choice_to_alt.get(label, '?')}" for label in sorted(choice_to_alt.keys()))}

Röstfördelning:
  Majoritet ({majority_count} votes): {majority_choice} — {majority_alt}
  Minoritet ({minority_count} votes): {minority_choice} — {minority_alt}

Din uppgift:
1. Identifiera det OLIKA ANTAGANDET som driver divergensen.
2. Klassificera divergenstypen: epistemic (vad är sant), normative (vad bör göras), scope (vilket område), eller cost_of_error (ej ett utan annat är farligare)
3. Föreslå integrationsstrategi: reframe (ändra perspektiv), tradeoff (acceptera båda), empirical_test (testa empiriskt), eller clarification (klargör begrepp)
4. Generera 3 konkreta nästa steg baserat på integration_mode

Svar i JSON-format (bara JSON, inget annat):
{{
  "majority_assumption": "Majoriteten antar att...",
  "minority_assumption": "Minoriteten antar att...",
  "divergence_source": "Divergensen kommer från antagandet om [X]",
  "divergence_type": "epistemic",
  "divergence_axes": [
    {{
      "dimension": "Namn på axel",
      "majority_position": 0.8,
      "minority_position": 0.2,
      "interpretation": "Vad denna axel betyder"
    }}
  ],
  "integration_strategy": "Konkret actionable strategi (inte bara narrativ)",
  "integration_mode": "clarification",
  "meta_question": "Nästa fråga vi bör ställa",
  "meta_alternatives": [
    "Alternativ 1: konkret nästa steg",
    "Alternativ 2: konkret nästa steg",
    "Alternativ 3: konkret nästa steg"
  ],
  "is_resolvable": true
}}
"""

    system = "Du är en filosofisk analytiker. Svara ENBART med giltigt JSON, inget annat. Fokus: operativ struktur, inte bara narrativ."

    # Use injected llm_fn, fallback to _call_llm
    llm_to_use = llm_fn if llm_fn else _call_llm
    if not llm_to_use:
        # Emergency fallback if no LLM available at all
        return _fallback_synthesize_reason(question, majority_choice, minority_choice, majority_alt, minority_alt, confidence)
    
    response_text = llm_to_use(system, prompt)

    if not response_text:
        # Fallback om LLM inte tillgänglig
        return _fallback_synthesize_reason(question, majority_choice, minority_choice, majority_alt, minority_alt, confidence)

    # Parsa JSON från response
    try:
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = {}
    except (json.JSONDecodeError, ValueError):
        data = {}

    # Return rich structured output
    return {
        'question': question,
        'majority_choice': majority_choice,
        'minority_choice': minority_choice,
        'majority_assumption': data.get('majority_assumption', f'Majoriteten föredrar {majority_choice}'),
        'minority_assumption': data.get('minority_assumption', f'Minoriteten föredrar {minority_choice}'),
        'divergence_source': data.get('divergence_source', 'Okänd'),
        'divergence_type': data.get('divergence_type', 'epistemic'),  # NEW: categorization
        'divergence_axes': data.get('divergence_axes', []),  # NEW: geometric structure
        'integration_strategy': data.get('integration_strategy', 'Kunde inte analysera'),
        'integration_mode': data.get('integration_mode', 'clarification'),  # NEW: categorization
        'meta_question': data.get('meta_question', 'Ingen metafråga genererad'),
        'meta_alternatives': data.get('meta_alternatives', []),  # NEW: dynamic alternatives
        'confidence': confidence,
        'is_resolvable': data.get('is_resolvable', True),
    }


def frame_transform(question: str, confidence: float = 0.0, llm_fn: Optional[Any] = None) -> dict:
    """
    Meta-nivå 2: Detektera om frågan själv är felställd.

    Returnerar:
      {
        'original_question': str,
        'is_well_framed': bool,
        'reframed_question': Optional[str],
        'problem_type': str,  # 'ill_posed' | 'ambiguous' | 'category_error' | 'ok'
        'recommendation': str,
      }
    """

    prompt = f"""
Fråga: {question}

Är denna fråga välställd för att kunna få ett klart svar?

Kontrollera:
1. Är termer tydligt definierade? (Eller är det begreppsförvirring?)
2. Kan frågan svaras objektivt? (Eller är det värdeomdöme presenterat som faktum?)
3. Finns det dolt antagande i frågeformuleringen? (Eller är den neutral?)

Om frågan är felställd, föreslå en OMFORMULERING.

Svar i JSON:
{{
  "is_well_framed": true/false,
  "problem_type": "ok" | "ill_posed" | "ambiguous" | "category_error",
  "reframed_question": "Omformulerad fråga eller null",
  "reason": "Förklaring kort och direkt"
}}
"""

    system = "Du är logiker. Svara ENBART med giltigt JSON."

    # Use injected llm_fn, fallback to _call_llm
    llm_to_use = llm_fn if llm_fn else _call_llm
    response_text = llm_to_use(system, prompt)

    if not response_text:
        return {
            'original_question': question,
            'is_well_framed': True,
            'reframed_question': None,
            'problem_type': 'ok',
            'recommendation': 'Kunde inte analysera (LLM ej tillgänglig)',
        }

    try:
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = {}
    except (json.JSONDecodeError, ValueError):
        data = {}

    return {
        'original_question': question,
        'is_well_framed': data.get('is_well_framed', True),
        'reframed_question': data.get('reframed_question'),
        'problem_type': data.get('problem_type', 'ok'),
        'recommendation': data.get('reason', 'Ingen analys tillgänglig'),
    }


def convergence_check(
    iteration: int,
    confidence_history: list[float],
    assumption_history: list[str],
    threshold: float = 0.05,
) -> dict:
    """
    Meta-nivå 3: Stoppa rekursion när systemet konvergerat.

    Input:
      iteration: Aktuell iteration
      confidence_history: C-värden från tidigare iterationer
      assumption_history: Extraherade huvudantaganden från tidigare iterationer
      threshold: Tillåten förändring innan stabil

    Output:
      {
        'should_continue': bool,
        'reason': str,
        'stability_score': float,  # 0-1, där 1 = perfekt stabil
      }
    """

    if iteration < 2:
        return {
            'should_continue': True,
            'reason': 'För tidigt att döma konvergens (< 2 iterationer)',
            'stability_score': 0.0,
        }

    # Kontrollera C-stabilitet
    recent_c = confidence_history[-2:]
    if len(recent_c) == 2:
        c_change = abs(recent_c[1] - recent_c[0])
        c_stable = c_change < threshold
    else:
        c_stable = False

    # Kontrollera antagandestabilitet
    recent_assumptions = assumption_history[-2:] if len(assumption_history) >= 2 else []
    if len(recent_assumptions) == 2:
        assumptions_same = recent_assumptions[0] == recent_assumptions[1]
    else:
        assumptions_same = False

    stability_score = (float(c_stable) + float(assumptions_same)) / 2.0

    should_continue = not (c_stable and assumptions_same)

    reason = ""
    if c_stable:
        reason += "✓ Confidence stabil. "
    else:
        reason += f"✗ Confidence varierar (Δ={c_change:.3f}). "

    if assumptions_same:
        reason += "✓ Antaganden stabila."
    else:
        reason += "✗ Antaganden har förändrats."

    return {
        'should_continue': should_continue,
        'reason': reason,
        'stability_score': stability_score,
    }


if __name__ == '__main__':
    # Demo på en enkelt divergens
    print("=" * 80)
    print("DIVERGENCE SEMANTICS — DEMO")
    print("=" * 80)

    result = synthesize_reason(
        question="Är hypotesen falsifierbar?",
        alternatives=[
            "A: Svag falsifierbarhet",
            "B: Delvis falsifierbar men kräver striktare mättrösklar",
            "C: Starkt falsifierbar med tydliga kriterier"
        ],
        vote_distribution={"B": 3, "C": 2},
        confidence=0.309,
        is_multimodal=False,
    )

    print("\n📊 RESULTAT")
    print(f"Fråga: {result['question']}")
    print(f"Majoritet: {result['majority_choice']} ({result['majority_assumption'][:60]}...)")
    print(f"Minoritet: {result['minority_choice']} ({result['minority_assumption'][:60]}...)")
    print(f"\n🔍 Divergence Source: {result['divergence_source'][:100]}...")
    print(f"🤝 Integration: {result['integration_strategy'][:100]}...")
    print(f"❓ Meta-question: {result['meta_question'][:100]}...")
    print(f"💪 Resolvable: {result['is_resolvable']}")
    print(f"📈 Confidence: {result['confidence']:.3f}")

    print("\n" + "=" * 80)
