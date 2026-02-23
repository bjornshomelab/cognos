# CognOS — Theoretical Insights and Empirical Findings

**Date:** February 21, 2026 (updated same day)  
**Status:** Active reference file — read by models, agents, and UI  
**Project:** `cognos-standalone`  
**Authors:** Björn Wikström (researcher), Claude (AI partner)

---

## What is CognOS?

CognOS is an epistemic integrity layer for agentic AI pipelines.

It is not a better way to generate answers. It is a verification layer that detects when an agent defaults to statistical best practice instead of context-grounded answers — and stops it before the error propagates.

**An agent without CognOS:**

```text
Input → Decision 1 (guesses) → Decision 2 (built on falsehood) → Decision 3 (error propagated) → Action
```

**An agent with CognOS:**

```text
Input → [CognOS] → Decision 1 verified → [CognOS] → Decision 2 verified → Action
```

It answers the question: *Does the agent know when it knows — and when it guesses?*

Jasper is the reference implementation. CognOS is the product. Everything else is window dressing.

The formula:

```text
C = p × (1 - Ue - Ua)

p   = model's prediction confidence [0, 1]
Ue  = epistemic uncertainty (variance of MC samples)
Ua  = aleatoric/semantic risk (ambiguity + irreversibility + blast_radius) / 3
C   = decision confidence [0, 1]
```

Four possible decisions based on C and Ue distribution:

- **auto** — C ≥ threshold, act autonomously
- **synthesize** — C low, bimodal Ue (perspective conflict → combine)
- **explore** — C low, unimodal Ue (noise → gather more data)
- **escalate** — high irreversibility AND low C (too risky)

Implemented in: `confidence.py` (v2), `jasper_cognos.py` (Jasper integration)

---

## Version History

| Version | Formula | Problem | Result |
| ------- | ------- | ------- | ------ |
| v1 | `C = p × (1-Ue)` | Missed overconfident errors | Safety Gain 0% |
| v1.5 | `C = p × (1-Ue-Ua)` with `Ua = 2p(1-p)` | Ua heuristic circular | Safety Gain 83% on synthetic data |
| v2 | Semantic Ua, multimodal Ue detection, four decisions | — | Operational in Jasper |

---

## Empirical Findings

### Finding 1: Validation on UCI Breast Cancer

- Dataset: 285 test samples, RandomForest (30 trees)
- Mean Ue: 0.036
- CognOS v1.5 wins at 40–55% escalation rate (+40–60% safety gain vs baseline)
- At >70% escalation: ceiling effect, all methods converge

**Implication:** CognOS is optimal for cost-constrained systems (limited human review capacity).

### Finding 2: Signal-Mismatch in Jasper Integration

When CognOS ran on narrative LLM responses (free text), it always gave EXPLORE, never SYNTHESIZE.

**Root cause:** Narrative responses collapse to *style similarity*, not *position similarity*.
Jaccard similarity on 300-word responses measures whether models write the same way — not whether they have the same opinion.

**Solution:** Structured choice format (see below).

### Finding 3: Diagnostic Experiment — Three Prompt Formats

The same question was run with three prompt formats yielding radically different results:

**Question:** "How should CognOS be triggered in decision chains — per question, per session, or at specific risk levels?"

| Prompt Type | Ue | Majority Answer |
| ----------- | -- | --------------- |
| Narrative (free text) | 0.005 | C: Risk levels |
| Forced binary (choice only) | 0.160 | A: Per question |
| Structured (choice + confidence) | 0.002 | B: Per session |

**Three prompt types. Three different answers. Ue differs 65×.**

---

## Theoretical Insights

### Insight 1: Prompt Format Shapes Answer, Not Just Measurement

Standard assumption in MC sampling for LLMs: temperature variation samples from the model's internal probability distribution.

**It does not.**

Prompt format fundamentally constrains the answer space. Change format → change majority answer. It is not measurement error. The answer changes.

Formally:

> *Standard MC sampling measures format-conditioned variance, not belief variance.*

### Insight 2: Taxonomy — Three Types of LLM Uncertainty

| Symbol | Name | Measures | Hidden? |
| ------ | ---- | -------- | ------- |
| U_model | Model uncertainty | Internal epistemic uncertainty | Partially |
| U_prompt | Prompt uncertainty | How much format constrains answer space | Yes |
| U_problem | Problem uncertainty | Question's intrinsic ill-posedness | Yes |

**Diagnostics:**

- `confidence = 0` in forced choice, but not in narrative → U_prompt
- `confidence = 0` in *all* three prompt types → U_problem (question is genuinely ill-formed)

### Insight 3: Confidence = 0 is a Signal about the Question, Not the Model

> *A zero-confidence outcome does not necessarily indicate uncertainty — it may signal that the decision frame is incompatible with the model's internal representation of the problem.*

This is missing from the literature.

### Insight 4: LLMs are Distribution Prisms, Not Mirrors

Common metaphor: LLMs as mirrors of human text.

More precisely: **prisms**. Incoming signal (the question) is refracted and transformed depending on the angle. Same model, same information, different prompt format → different spectrum of answers.

The model has no stable internal opinion. It constructs an answer matching the pattern "this type of question → this type of answer" in training data.

### Insight 5: Frame Sensitivity — Human vs AI

Framing effects exist in humans too (Kahneman). It is not AI-unique.

**The difference:** Humans can *know about it* — and compensate via metacognition.
The model has no access to its own frame sensitivity.

> *Uncertainty in LLMs is not a property of the model alone. It is a property of how the question is formed.*

### Insight 6: CognOS as an Epistemological Layer

CognOS is not intelligence. It is not reasoning.

It is the external compensation mechanism that the model lacks internally — a system that asks:

> *"Is this answer stable, or is it an artifact of how we asked?"*

It is an epistemological layer on top of a statistical system.

That is what is new.

### Insight 7: Context Memory is Necessary but Not Sufficient

An LLM with access to context does not guarantee context-grounded answers. Without verification, the model can read the project file and still respond with generic best practice.

Empirical evidence (2026-02-21): the same questions were run without and with `INSIGHTS.md` as system context. All three majority answers changed. Without context → generic AI research best practice. With context → situation-adapted answers based on actual findings.

Three layers are required:

- **Memory** — context is available
- **Grounding** — answer is based on context
- **Verification** — CognOS checks that grounding is correct

> *An LLM with access to context does not guarantee context-grounded responses — it requires a verification layer to detect when the model defaults to statistical best practice despite available context.*

### Insight 9: mHC som klassisk kvant-gate-analog — och CognOS mätpunkten

**Datum:** 2026-02-23
**Källa:** Observation av Björn Wikström vid genomläsning av arXiv:2512.24880v2

mHC:s residualmapping H_i^res är en **dubbelt stokastisk matris** (Birkhoff-polytopen). Birkhoff-von Neumann-teoremet säger att dessa är konvexa kombinationer av permutationsmatriser. Permutationsmatriser är de *klassiska* analogerna till kvanttransformationer (unitära matriser).

Strukturellt är mHC därmed en **klassisk kvant-gate-arkitektur**:

```text
[kvant-gate]   = unitär matris på qubitar i superposition
[mHC H_i^res]  = dubbelt stokastisk matris på n parallella strömmar
```

Samma topologi. Klassisk sannolikhet istället för kvantamplitud.

**Implikationen för CognOS:**

I kvantdatorer kollapsar mätning superpositionstillståndet. I mHC kollapsar H_i^post de n strömmarna tillbaka till ett enda layer-output. Det fönster som existerar *precis före* H_i^post-kollaps är CognOS naturliga mätpunkt — osäkerheten är maximal och geometriskt separerbar i detta fönster.

```text
[ström 1]  ─┐
[ström 2]   ├─ [H_i^post kollaps] → x_{l+1}
[ström 3]   │       ↑
[ström 4]  ─┘   CognOS mäter här
```

**Varför detta är nytt:** Nuvarande CognOS mäter post-hoc på output. mHC öppnar ett pre-kollaps-fönster — ett strukturellt analogt med kvantobestämdhet precis före mätning. Det är en arkitektoniskt inbyggd epistemisk signal, inte en approximation.

**Koppling till HYPOTHESIS 2:** Se `HYPOTHESIS.md` — Signal 1–3 operationaliserar detta fönster.

### Insight 8: CognOS is What is Missing in Agentic AI

Everyone builds agents. No one has solved that agents do not know when they are guessing.

The problem is not that agents are stupid. It is that they lack epistemic honesty — they present guesses with the same confidence as well-grounded answers.

CognOS gives the agent a word for what it does not know:

- `auto` — I know, acting
- `explore` — I am uncertain, need more
- `synthesize` — I hold two conflicting perspectives, combine them
- `escalate` — this is too risky, human needed

It is not safety in the technical sense. It is cognitive integrity.

CognOS ships as a standalone function — `pip install cognos` — that any agent, model, or pipeline can import.

---

## Structured Choice — Solution to Signal-Mismatch

For CognOS to measure actual consensus, the model must report discrete positions.

**Prompt Format:**

```text
CHOICE: <A/B/C>
CONFIDENCE: <0.0–1.0>
RATIONALE: <max 20 words>
```

**Results from Three Real Decisions (2026-02-21):**

| Question | C | Decision | Votes |
| -------- | - | -------- | ----- |
| Paper: agentic AI vs classical ML? | 0.894 | AUTO | 5/5 → Agentic AI |
| PhD: papers vs contacts? | 0.998 | AUTO | 5/5 → Contacts |
| CognOS trigger design? | 0.445 | SYNTHESIZE ⊕ | 4/5 C, 1/5 B |

SYNTHESIZE triggered when 4/5 models chose one direction but 1 chose another — perspective conflict correctly detected.

Implemented in: agentic frameworks and decision systems.

---

## Publication Contribution (CognOS Paper)

**Primary contribution:**
Epistemic + aleatoric uncertainty combined: `C = p × (1 - Ue - Ua)`

**Secondary contribution (newly discovered):**

> *CognOS requires decision-structured prompts to reveal consensus geometry. Free-form responses collapse onto narrative similarity, masking real epistemic divergence between model perspectives.*

**Tertiary contribution (most original):**

> *MC sampling on LLMs does not measure the model's internal uncertainty. It measures how much the prompt format constrains the answer space.*

**Positioning:**

- Target audience: cost-constrained systems with 40–60% escalation budget
- Not for: high-stakes systems that escalate >80% (ceiling effect)

---

## Files

| File | Content |
| ---- | ------- |
| `cognos/confidence.py` | CognOS v2 — formula, multimodal detection, four decisions |
| `cognos/divergence_semantics.py` | Layer 2–3 — divergence extraction & convergence control |
| `cognos/core/cognos_deep.py` | Five-layer recursive analysis stack |
| `cognos/core/cognos_integration_demo.py` | Working example of all three layers |
| `experiments/eval_hypothesis.py` | Run CognOS on HYPOTHESIS |
| `research/HYPOTHESIS.md` | Operational definitions & falsification criteria |
| `research/INSIGHTS.md` | This file |

---

## Next Steps

### Immediately (Step 1)

1. **Hypothesis:** Operationalize U_model/U_prompt/U_problem with clear definitions and falsification criteria → HYPOTHESIS.md

### Empirical Validation (Step 2)

1. **10 × 3 × 3 Experiment:** 10 questions × 3 types (factual/decision/value) × 3 prompt formats = 90 runs
   - API: Groq (reproducible, open)
   - Output: validation results table + three figures

### Paper (Step 3)

1. **Four Agentic Scenarios:** Show CognOS in chains, not single decisions
   - Scenario 1: Research summary (5 papers → divergence detected)
   - Scenario 2: Multi-step decision (error in step 1 propagates to step 4)
   - Scenario 3: Knowledge-intensive question (U_model vs U_prompt)
   - Scenario 4: Competing papers (hidden conflicts via SYNTHESIZE signal)
2. **Target Conference:** NeurIPS / ICML / ACL (ML Safety / Alignment track)

### Open Source (Step 4)

1. **Standalone Package:** `pip install cognos` — zero Jasper dependencies
2. **GitHub:** Applied-Ai-Philosophy/cognos — README + working examples
3. **Documentation:** Clarity + integration guides for other frameworks

---

*This file is written to be readable by a new model or agent without background context. All central insights should be self-explanatory.*
