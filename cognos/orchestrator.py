#!/usr/bin/env python3
"""
cognos_orchestrator.py — Full epistemic orchestration pipeline.

Connects:
  1. Structured choice voting (MC sampling)
  2. Confidence calculation (Layer 1)
  3. Assumption extraction (Layer 2) 
  4. Convergence detection (Layer 3)
  5. Recursive meta-reasoning (Layers 4+)

This is the "cognitive orchestration" — the actual thinking process.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Callable, Any

sys.path.insert(0, str(Path(__file__).parent))

from confidence import compute_confidence
from divergence_semantics import synthesize_reason, convergence_check, frame_transform
try:
    from strong_synthesis import synthesize_strong, compute_epistemic_gain
    STRONG_SYNTHESIS_AVAILABLE = True
except ImportError:
    STRONG_SYNTHESIS_AVAILABLE = False
    synthesize_strong = None


def ask_llm_groq(system: str, prompt: str, temperature: float = 0.7) -> Optional[str]:
    """Call Groq API directly (no Jasper dependency)."""
    try:
        from groq import Groq
        client = Groq()
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠ Groq call failed: {e}", file=sys.stderr)
        return None


def parse_structured_choice(text: str, labels: list[str]) -> tuple[Optional[str], float, str]:
    """
    Parse CHOICE + CONFIDENCE + RATIONALE from response.
    
    Robust to multi-line rationale and extra whitespace.
    Returns: (choice_label, confidence_score, rationale)
    """
    import re
    choice = None
    confidence = 0.5
    rationale_lines = []
    mode = None
    
    for raw in text.splitlines():
        line = raw.strip()
        u = line.upper()
        
        if u.startswith("CHOICE:"):
            mode = "choice"
            val = line.split(":", 1)[1].strip().upper()
            # Match strictly: startswith, not "in"
            for label in labels:
                if val.startswith(label):
                    choice = label
                    break
        
        elif u.startswith("CONFIDENCE:"):
            mode = "conf"
            m = re.search(r"[\d.]+", line)
            if m:
                try:
                    confidence = float(m.group())
                except:
                    pass
            confidence = min(1.0, max(0.0, confidence))
        
        elif u.startswith("RATIONALE:"):
            mode = "rat"
            rationale_lines.append(line.split(":", 1)[1].strip() if ":" in line else "")
        
        elif mode == "rat" and line:
            # Collect multi-line rationale
            rationale_lines.append(line)
    
    rationale = " ".join(rationale_lines).strip()
    return choice, confidence, rationale


class CognOSOrchestrator:
    """
    Full epistemic reasoning loop.
    
    Orchestrates voting → confidence → divergence → convergence → recursion.
    """
    
    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        max_depth: int = 4,
        convergence_threshold: float = 0.05,
        confidence_threshold: float = 0.72,
        multimodal_threshold: float = 0.20,
        verbose: bool = True,
    ):
        self.llm_fn = llm_fn or ask_llm_groq
        self.max_depth = max_depth
        self.convergence_threshold = convergence_threshold
        self.confidence_threshold = confidence_threshold
        self.multimodal_threshold = multimodal_threshold
        self.verbose = verbose
    
    def _log(self, msg: str, level: str = "info"):
        if self.verbose:
            colors = {
                "info": "\033[90m",
                "success": "\033[32m",
                "warning": "\033[33m",
                "error": "\033[31m",
            }
            reset = "\033[0m"
            color = colors.get(level, "\033[90m")
            print(f"{color}[CognOS] {msg}{reset}")
    
    def _is_multimodal(self, confidences: list[float]) -> bool:
        """Detect bimodal vs unimodal distribution via cluster separation."""
        if len(confidences) < 3:
            return False
        
        sorted_conf = sorted(confidences)
        mid = len(sorted_conf) // 2
        
        low_group = sorted_conf[:mid]
        high_group = sorted_conf[mid:]
        
        if not low_group or not high_group:
            return False
        
        low_mean = sum(low_group) / len(low_group)
        high_mean = sum(high_group) / len(high_group)
        separation = high_mean - low_mean
        
        return separation > self.multimodal_threshold
    
    def _strip_label_prefix(self, s: str) -> str:
        """Remove leading 'A: ', 'B: ', etc. from alternative text."""
        s = s.strip()
        if len(s) >= 2 and s[1] == ":" and s[0].upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return s[2:].strip()
        return s
    
    def run_structured_choice(
        self,
        question: str,
        alternatives: list[str],
        context: str = "",
        n_samples: int = 5,
    ) -> dict:
        """
        Run MC voting with structured choice format.
        
        Returns: {decision, confidence, epistemic_ue, votes, majority_choice, ...}
        """
        self._log(f"Voting: {question[:70]}...")
        
        labels = [chr(65 + i) for i in range(len(alternatives))]
        # Sanera alternatives — ta bort dubblad "A: "
        clean_alts = [self._strip_label_prefix(alt) for alt in alternatives]
        alt_text = "\n".join(f"{la}: {alt}" for la, alt in zip(labels, clean_alts))
        
        prompt = f"""Question: {question}

Alternatives:
{alt_text}

Answer ONLY in this format:
CHOICE: <{'/'.join(labels)}>
CONFIDENCE: <0.0-1.0>
RATIONALE: <max 20 words>"""
        
        system = (
            f"You are a decision analyst. Answer in exactly the specified format.\n\n"
            f"=== CONTEXT ===\n{context}"
            if context else
            "You are a decision analyst. Answer in exactly the specified format."
        )
        
        # Collect n_samples
        choices, confidences = [], []
        for i in range(n_samples):
            response = self.llm_fn(system, prompt)
            if response:
                choice, conf, _ = parse_structured_choice(response, labels)
                if choice:
                    choices.append(choice)
                    confidences.append(conf)
        
        if not choices:
            self._log(f"No valid responses from LLM", "error")
            return {
                'decision': 'escalate',
                'confidence': 0.0,
                'epistemic_ue': 1.0,
                'aleatoric_ua': 0.0,
                'votes': {},
                'majority_choice': None,
                'samples': 0,
            }
        
        # Count votes
        votes = {la: choices.count(la) for la in labels if la in choices}
        majority = max(votes, key=votes.get)
        p = votes[majority] / len(choices)
        
        # MC predictions: probability that majority choice is correct
        # Use mean confidence of majority votes + vote proportion as signal
        maj_confs = [cf for ch, cf in zip(choices, confidences) if ch == majority]
        mean_maj_conf = sum(maj_confs) / len(maj_confs) if maj_confs else p
        
        # Combine: majority frequency + quality of majority votes
        combined_signal = 0.6 * p + 0.4 * mean_maj_conf
        mc_predictions = [combined_signal] * max(4, len(choices))
        
        # Add some variance from individual confidence variation
        if len(maj_confs) >= 2:
            for i in range(min(len(mc_predictions), len(maj_confs))):
                mc_predictions[i] = maj_confs[i]
        
        result = compute_confidence(p, mc_predictions)
        result['votes'] = votes
        result['majority_choice'] = majority
        result['majority_label'] = clean_alts[labels.index(majority)]
        result['samples'] = len(choices)
        result['is_multimodal'] = self._is_multimodal(confidences)
        
        # Normalize key names for consistency
        result['epistemic_ue'] = result.get('epistemic_uncertainty', result.get('epistemic_ue', 0.0))
        result['aleatoric_ua'] = result.get('aleatoric_uncertainty', result.get('aleatoric_ua', 0.0))
        
        # CRITICAL: Force SYNTHESIZE if multimodal (perspective conflict signal)
        if result['is_multimodal']:
            result['decision'] = 'synthesize'
            self._log(f"  ⚠ Multimodal detected → Forcing SYNTHESIZE", "warning")
        
        self._log(f"  → {majority} ({p:.0%}), C={result['confidence']:.3f}, "
                 f"decision={result['decision'].upper()}")
        
        return result
    
    def orchestrate(
        self,
        question: str,
        alternatives: list[str],
        context: str = "",
        n_samples: int = 5,
    ) -> dict:
        """
        Full orchestration: voting → confidence → divergence → convergence → recursion.
        
        Returns: {
            decision,
            confidence,
            final_answer,
            layers: [{iteration, question, decision, confidence, assumptions, ...}],
            converged,
            meta_history: [assumptions by iteration]
        }
        """
        self._log(f"Starting orchestration: {question[:70]}...")
        
        # Layer 0: Frame validation (check if question is well-posed)
        frame_check = frame_transform(question, llm_fn=self.llm_fn)
        if not frame_check['is_well_framed']:
            self._log(f"  ⚠ Ill-posed question: {frame_check['problem_type']}", "warning")
            if frame_check['reframed_question']:
                self._log(f"  Suggestion: {frame_check['reframed_question']}", "warning")
                question = frame_check['reframed_question']
        
        layers = []
        confidence_history = []
        assumption_history = []
        
        current_question = question
        current_alternatives = alternatives
        
        for depth in range(self.max_depth):
            self._log(f"\n{'='*70}")
            self._log(f"LAYER {depth + 1}/{self.max_depth}")
            self._log(f"{'='*70}")
            
            # Layer 1: Voting & confidence
            vote_result = self.run_structured_choice(
                current_question,
                current_alternatives,
                context,
                n_samples
            )
            
            C = vote_result['confidence']
            decision = vote_result['decision']
            confidence_history.append(C)
            
            layer_dict = {
                'iteration': depth + 1,
                'question': current_question,
                'alternatives': current_alternatives,
                'decision': decision,
                'confidence': C,
                'epistemic_ue': vote_result['epistemic_ue'],
                'aleatoric_ua': vote_result['aleatoric_ua'],
                'votes': vote_result['votes'],
                'majority_choice': vote_result['majority_choice'],
                'majority_label': vote_result.get('majority_label'),
                'is_multimodal': vote_result.get('is_multimodal', False),
            }
            
            # Early exit if high confidence
            if decision == 'auto':
                self._log(f"✓ HIGH CONFIDENCE → AUTO decision", "success")
                layer_dict['reasoning'] = "Confidence sufficient, acting autonomously"
                layers.append(layer_dict)
                return {
                    'decision': decision,
                    'confidence': C,
                    'final_answer': vote_result.get('majority_label'),
                    'layers': layers,
                    'converged': True,
                    'iterations': len(layers),
                }
            
            # Layer 2: Divergence extraction (if SYNTHESIZE)
            if decision == 'synthesize':
                self._log(f"✨ DIVERGENCE DETECTED → Running strong synthesis...", "warning")
                
                # Sanitize alternatives for divergence analysis
                clean_curr_alts = [self._strip_label_prefix(a) for a in current_alternatives]
                
                # Use strong synthesis if available, fallback to weak
                if STRONG_SYNTHESIS_AVAILABLE and synthesize_strong:
                    divergence = synthesize_strong(
                        question=current_question,
                        alternatives=clean_curr_alts,
                        vote_distribution=vote_result['votes'],
                        confidence=C,
                        context=context,
                        llm_fn=self.llm_fn,
                    )
                    assumption_history.append(divergence['analysis']['core_tension'])
                else:
                    # Fallback to weak synthesis (now returns rich structure)
                    divergence = synthesize_reason(
                        question=current_question,
                        alternatives=clean_curr_alts,
                        vote_distribution=vote_result['votes'],
                        confidence=C,
                        is_multimodal=vote_result.get('is_multimodal', False),
                        context=context,
                        llm_fn=self.llm_fn,
                    )
                
                layer_dict['divergence'] = divergence
                
                # Log key information from divergence (both strong and weak format)
                self._log(f"  Majority: {divergence.get('majority_assumption', 'Unknown')[:50]}...")
                self._log(f"  Minority: {divergence.get('minority_assumption', 'Unknown')[:50]}...")
                self._log(f"  Type: {divergence.get('divergence_type', 'unknown')} | Mode: {divergence.get('integration_mode', 'unknown')}")
                
                if divergence.get('divergence_axes'):
                    self._log(f"  Geometry: {len(divergence['divergence_axes'])} dimensions")
                
                if divergence.get('meta_alternatives'):
                    self._log(f"  Next steps: {len(divergence['meta_alternatives'])} options")
                    self._log(f"    → {divergence['meta_alternatives'][0][:50]}..." if divergence['meta_alternatives'] else "")
                    assumption_history.append(divergence.get('divergence_source', 'unknown'))
                
                # Layer 3: Convergence check
                if len(confidence_history) >= 2:
                    conv = convergence_check(
                        len(confidence_history),
                        confidence_history,
                        assumption_history,
                        self.convergence_threshold
                    )
                    self._log(f"  Convergence: {conv['reason']}")
                    
                    if not conv['should_continue']:
                        self._log(f"✓ CONVERGED → Stopping recursion", "success")
                        layers.append(layer_dict)
                        return {
                            'decision': decision,
                            'confidence': C,
                            'final_answer': divergence.get('integration_strategy'),
                            'layers': layers,
                            'converged': True,
                            'iterations': len(layers),
                            'reason': conv['reason'],
                        }
                
                # Prepare next iteration with meta-alternatives
                meta_q = None
                if isinstance(divergence, dict) and 'meta_alternatives' in divergence:
                    meta_q = divergence['meta_alternatives'].get('meta_question')
                    meta_alts = divergence['meta_alternatives'].get('alternatives', [])
                elif isinstance(divergence, dict):
                    meta_q = divergence.get('meta_question')
                
                if meta_q:
                    self._log(f"  → Meta-question: {meta_q[:70]}...")
                    current_question = meta_q
                    
                    # Use generated alternatives if available
                    if isinstance(divergence, dict) and 'meta_alternatives' in divergence:
                        meta_alts = divergence['meta_alternatives'].get('alternatives', [])
                        if meta_alts:
                            current_alternatives = [
                                f"{i+1}. {alt.get('action', str(alt))}"
                                for i, alt in enumerate(meta_alts[:3])
                            ]
                        else:
                            current_alternatives = [
                                "1. Prioritize conceptual clarity",
                                "2. Focus on empirical precision",
                                "3. Assess practical implications",
                            ]
                    else:
                        current_alternatives = [
                            "Perspective A stronger",
                            "Perspective B stronger",
                            "Equivalent weight",
                        ]
                else:
                    self._log(f"  (no meta-question; stopping recursion)", "warning")
                    break
            
            # Layer 1.5: Continue on EXPLORE
            elif decision == 'explore':
                self._log(f"? NOISE DETECTED → Need more information", "warning")
                layer_dict['reasoning'] = "Noise (unimodal low Ue) — would need more context"
                layers.append(layer_dict)
                break
            
            # Layer 1.5: Escalate
            elif decision == 'escalate':
                self._log(f"⚠ HIGH RISK → Escalating to human", "error")
                layer_dict['reasoning'] = "Risk too high for autonomous decision"
                layers.append(layer_dict)
                break
            
            layers.append(layer_dict)
        
        # Return final result
        return {
            'decision': layers[-1]['decision'] if layers else 'escalate',
            'confidence': confidence_history[-1] if confidence_history else 0.0,
            'final_answer': layers[-1].get('majority_label') if layers else None,
            'layers': layers,
            'converged': False,
            'iterations': len(layers),
        }


def main():
    """Demo: Run on three bedömning questions."""
    
    orchestrator = CognOSOrchestrator(verbose=True)
    
    context = """CognOS is an epistemic integrity layer for AI systems.
    
Formula: C = p × (1 - Ue - Ua)

Three uncertainty types:
- U_model: internal epistemic uncertainty
- U_prompt: format-induced uncertainty  
- U_problem: ill-posedness of the question itself

The HYPOTHESIS_V02 proposes these are separable and testable."""
    
    questions = [
        {
            "question": "How falsifiable is HYPOTHESIS_V02 in its current form?",
            "alternatives": [
                "A: Weakly falsifiable",
                "B: Partially falsifiable but requires stricter thresholds",
                "C: Strongly falsifiable with clear criteria"
            ]
        },
        {
            "question": "What is HYPOTHESIS_V02's most critical weakness right now?",
            "alternatives": [
                "A: Empirical foundation too narrow",
                "B: U_prompt and U_model overlap too much",
                "C: Decision rule is underspecified"
            ]
        },
        {
            "question": "Should we run the 10×3×3 validation experiment now?",
            "alternatives": [
                "A: Yes, definitions are sufficient",
                "B: Yes, but sharpen definitions first",
                "C: No, need more preliminary work"
            ]
        }
    ]
    
    results = []
    
    for i, q in enumerate(questions, 1):
        print(f"\n\n{'#'*80}")
        print(f"# QUESTION {i}/3")
        print(f"{'#'*80}\n")
        
        result = orchestrator.orchestrate(
            question=q['question'],
            alternatives=q['alternatives'],
            context=context,
            n_samples=3,  # Reduced for demo speed
        )
        
        results.append(result)
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Final Decision: {result['decision'].upper()}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Iterations: {result['iterations']}")
        print(f"Converged: {'Yes' if result['converged'] else 'No'}")
        print(f"Answer: {result['final_answer']}")
    
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    for i, (q, r) in enumerate(zip(questions, results), 1):
        print(f"\n{i}. {q['question'][:70]}...")
        print(f"   Decision: {r['decision'].upper()}")
        print(f"   Confidence: {r['confidence']:.3f} / 1.0")
        print(f"   Answer: {r['final_answer']}")


if __name__ == '__main__':
    main()
