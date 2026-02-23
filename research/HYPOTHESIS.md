# HYPOTHESIS — Operationalization of Uncertainty Sources

**Date:** February 21, 2026  
**Status:** Candidate hypothesis for empirical testing (10×3×3)  
**Source:** Björn Wikström with Claude AI

## Core Hypothesis

LLM uncertainty in decision questions is a mixture of three separable components:
- **U_model** (internal epistemic uncertainty)
- **U_prompt** (format-induced uncertainty)
- **U_problem** (intrinsic ill-posedness of the question)

CognOS should be able to distinguish these sufficiently well to improve decision gating compared to prediction probability alone.

## Operational Definitions

- **U_model:** Variance that persists when prompt format is held constant but sampling varies (temperature/seed). Measure: within-format variance in responses/confidence.

- **U_prompt:** Variance that arises when question + model are constant but format changes (narrative, forced binary, structured choice). Measure: between-format difference in majority choice and/or Ue.

- **U_problem:** Stable high uncertainty regardless of format. Measure: low confidence in all three formats + no robust majority.

## Predictions

1. The same question exhibits significantly larger between-format variance than within-format variance for a non-trivial proportion of questions (U_prompt exists).

2. Structured choice reduces measurement artifacts and yields more stable consensus geometry than free text.

3. Questions with genuine ambiguity display persistent low-confidence outcomes in all formats (U_problem).

## Falsification Criteria

The hypothesis is rejected wholly or partially if any of the following hold in the 10×3×3 experiment:

1. **No format sensitivity:** between-format variance ≈ within-format variance for nearly all questions.

2. **No choice instability:** majority choice remains unchanged across formats in nearly all cases.

3. **No robust U_problem signal:** low confidence does not appear consistently across all formats for ill-posed questions.

## Decision Rule for CognOS

- Dominant between-format variance ⇒ flag **U_prompt risk**.
- Persistent low-confidence outcomes across formats ⇒ flag **U_problem risk**.
- Low within-format stability under fixed format ⇒ flag **U_model risk**.

## Experiment Design

**10×3×3 Matrix:**

| Dimension | Values | Purpose |
|-----------|--------|---------|
| Questions | 10 diverse decision questions | Coverage across domains |
| Formats | Narrative, Forced Binary, Structured Choice | Isolate format effects |
| Repetitions | 3 samples per format | Measure within-format variance |

**Metrics:**

- `Ue` (epistemic uncertainty): variance of MC predictions
- `p_format_stability`: proportion of questions where majority choice stable across formats
- `C_consistency`: consistency of confidence scores across formats

**Success Criterion:**

- ≥70% of questions show p_format_stability = False
- ≥60% of ill-posed questions show Ue_problem > 2 × Ue_well-posed

---

## HYPOTHESIS 2 — CognOS som nativ epistemisk signal i mHC-arkitektur

**Datum:** 2026-02-23
**Status:** Spekulativ — kräver arkitektonisk integration
**Källa:** Björn Wikström med Claude AI, baserat på DeepSeek mHC-paper (arXiv:2512.24880v2)

## Bakgrund

Manifold-Constrained Hyper-Connections (mHC) utökar residual-strömmen till n parallella strömmar per lager och projicerar blandningsmatrisen H_i^res på Birkhoff-polytopen via Sinkhorn-Knopp. Detta skapar tre geometriska strukturer som är epistemiskt meningsfulla.

## Kärnhypotes

mHC-arkitekturen exponerar tre nativa epistemiska signaler som CognOS-instanser kan mäta direkt, utan post-hoc-approximation:

### Signal 1: Inter-ström-divergens

Med n=4 parallella strömmar per lager representeras samma input i fyra separata representationer. Divergensen mellan dessa strömmar är ett direkt mått på epistemisk osäkerhet vid det lagret.

- Hög konvergens mellan strömmar → modellen är "enig med sig själv" → låg U_model
- Hög divergens → representationerna pekar åt olika håll → hög U_model

### Signal 2: H_i^res blandningsentropin

H_i^res är en dubbelt stokastisk matris (konvex kombination av permutationer). Entropin i matrisen mäter hur nära den är en ren permutation vs ett jämnt genomsnitt:

- Nära permutation (låg entropi) = ordnat informationsflöde = epistemisk säkerhet
- Nära uniform (hög entropi) = blandad, diffus routing = epistemisk osäkerhet

Formellt: `H_entropy = -sum(H_i^res * log(H_i^res))`

### Signal 3: H_i^pre routing-koncentration

H_i^pre aggregerar n strömmar till ett layer-input. Koncentrationen (t.ex. Gini-koefficient) mäter om modellen "vet vilken representation den ska lita på".

- Koncentrerad routing → en ström dominerar → säker aggregering
- Flat routing → alla strömmar väger lika → modellen vet inte vilken att lita på

## Prediktioner

1. Inter-ström-divergens korrelerar med U_model i befintliga CognOS-experiment.

2. Lager med hög H_i^res-entropi producerar svar med högre format-känslighet (U_prompt).

3. Kombinationen av alla tre signaler ger bättre epistemisk separabilitet än enbart logit-baserade konfidensestimat.

## Implikation för utopi-arkitekturen

Om mHC adopteras som standard i framtida modeller kan CognOS integreras *i* forward pass istället för som ett separat mätlager. Varje token-generation producerar då automatiskt:

```text
svar: X
motivering: "ström 2-3 konvergerade (divergens=0.04), routing koncentrerad (Gini=0.78)"
alternativ: Y avvisades pga hög inter-ström-divergens i lager 18-22
```

Detta är den tekniska grunden för modeller som kan säga "jag föreslår X istället för Y baserat på Z".

## Koppling till befintlig CognOS-teori

- U_model (Hypotes 1) ↔ Inter-ström-divergens (Hypotes 2, Signal 1)
- U_prompt (Hypotes 1) ↔ H_i^res-entropi (Hypotes 2, Signal 2)
- Routing-koncentration är en ny signal utan direkt motsvarighet i Hypotes 1

## Nästa steg

1. Verifiera om inter-ström-divergens kan approximeras i befintliga modeller utan mHC
2. Vänta på att mHC-kompatibla modeller blir tillgängliga för experiment
3. Diskutera med Naimat Ullah: hans UI:s "Intervention State" matchar Signal 1-3 som trigger
