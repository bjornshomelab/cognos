#!/usr/bin/env python3
"""
assumption_extraction.py — Explicit latent assumption detection engine.

While synthesize_reason() extracts assumptions from voting divergence,
this module provides structured extraction capabilities:

1. Latent assumptions detection — what is silently assumed but not stated
2. Perspective mapping — which ontological/epistemological frame each view uses
3. Ontology tagging — which entities/categories are presupposed

This is the "assumption extraction engine" that Björn identified as missing.
"""

import json
import re
from typing import Optional, Any
from dataclasses import dataclass
from enum import Enum


class AssumptionType(Enum):
    """Types of assumptions that can be detected."""
    ONTOLOGICAL = "ontological"  # About what exists
    EPISTEMOLOGICAL = "epistemological"  # About what can be known
    NORMATIVE = "normative"  # About what should be
    METHODOLOGICAL = "methodological"  # About how to investigate
    DEFINITIONAL = "definitional"  # About what terms mean
    SCOPE = "scope"  # About what domain we're in


class PerspectiveFrame(Enum):
    """Epistemological frames that perspectives can adopt."""
    EMPIRICIST = "empiricist"  # Truth through observation
    RATIONALIST = "rationalist"  # Truth through logic
    PRAGMATIST = "pragmatist"  # Truth through consequences
    CONSTRUCTIVIST = "constructivist"  # Truth through social construction
    REALIST = "realist"  # Truth exists independently
    INSTRUMENTALIST = "instrumentalist"  # Truth is usefulness


@dataclass
class LatentAssumption:
    """A single latent assumption extracted from text."""
    assumption_text: str
    assumption_type: AssumptionType
    confidence: float  # 0-1: How confident are we this is assumed?
    is_explicit: bool  # False = latent, True = explicitly stated
    supporting_evidence: str  # Text fragment that reveals this assumption


@dataclass
class PerspectiveMapping:
    """Maps a perspective to its epistemological/ontological frame."""
    perspective_label: str  # e.g., "Alternative A"
    perspective_text: str
    epistemological_frame: PerspectiveFrame
    ontological_commitments: list[str]  # What entities does this view assume exist?
    methodological_commitments: list[str]  # What methods does it prioritize?
    frame_confidence: float  # 0-1: How confident is this classification?


@dataclass
class OntologyTag:
    """Marks which entities/categories are presupposed."""
    entity: str  # e.g., "consciousness", "uncertainty", "truth"
    category: str  # e.g., "mental_state", "epistemic_property", "abstract_object"
    presupposed_by: list[str]  # Which alternatives presuppose this?
    is_contested: bool  # Do different alternatives disagree about its existence?


def extract_latent_assumptions(
    question: str,
    alternatives: list[str],
    context: Optional[str] = None,
    llm_fn: Optional[Any] = None,
) -> list[LatentAssumption]:
    """
    Extract latent assumptions from a question and its alternatives.
    
    Returns: List of LatentAssumption objects
    """
    
    if not llm_fn:
        return []  # Cannot extract without LLM
    
    prompt = f"""Analyze this question and alternatives for LATENT (unstated) assumptions.

Question: {question}

Alternatives:
{chr(10).join(f'{i+1}. {alt}' for i, alt in enumerate(alternatives))}

Context: {context or 'None'}

Identify assumptions that are IMPLICIT but NOT EXPLICITLY STATED.

For each assumption, determine:
- What is being assumed?
- What TYPE of assumption? (ontological | epistemological | normative | methodological | definitional | scope)
- How confident are you it's assumed? (0.0-1.0)
- Which text fragment reveals it?

Return JSON array:
[
  {{
    "assumption_text": "X is assumed to exist/be true",
    "assumption_type": "ontological",
    "confidence": 0.85,
    "is_explicit": false,
    "supporting_evidence": "Text fragment that reveals this"
  }},
  ...
]

Return ONLY valid JSON array, no other text.
"""
    
    system = "You are a philosophical analyst specializing in assumption detection."
    
    response = llm_fn(system, prompt)
    if not response:
        return []
    
    try:
        # Extract JSON array
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            return []
        
        # Parse into LatentAssumption objects
        assumptions = []
        for item in data:
            try:
                assumption = LatentAssumption(
                    assumption_text=item.get('assumption_text', 'Unknown'),
                    assumption_type=AssumptionType(item.get('assumption_type', 'epistemological')),
                    confidence=float(item.get('confidence', 0.5)),
                    is_explicit=bool(item.get('is_explicit', False)),
                    supporting_evidence=item.get('supporting_evidence', ''),
                )
                assumptions.append(assumption)
            except (ValueError, KeyError):
                continue  # Skip malformed entries
        
        return assumptions
    
    except (json.JSONDecodeError, ValueError):
        return []


def map_perspectives(
    alternatives: list[str],
    context: Optional[str] = None,
    llm_fn: Optional[Any] = None,
) -> list[PerspectiveMapping]:
    """
    Map each alternative to its epistemological and ontological frame.
    
    Returns: List of PerspectiveMapping objects
    """
    
    if not llm_fn:
        return []
    
    prompt = f"""Analyze these alternatives and identify their epistemological/ontological frames.

Alternatives:
{chr(10).join(f'{i+1}. {alt}' for i, alt in enumerate(alternatives))}

Context: {context or 'None'}

For each alternative, determine:
1. Epistemological frame: empiricist | rationalist | pragmatist | constructivist | realist | instrumentalist
2. Ontological commitments: What entities does this view assume exist?
3. Methodological commitments: What methods does it prioritize?
4. Confidence: How confident are you in this classification? (0.0-1.0)

Return JSON array:
[
  {{
    "perspective_label": "Alternative 1",
    "perspective_text": "Full text of alternative",
    "epistemological_frame": "empiricist",
    "ontological_commitments": ["observable phenomena", "causal relations"],
    "methodological_commitments": ["controlled experiments", "measurement"],
    "frame_confidence": 0.80
  }},
  ...
]

Return ONLY valid JSON array.
"""
    
    system = "You are a philosophical analyst specializing in perspective classification."
    
    response = llm_fn(system, prompt)
    if not response:
        return []
    
    try:
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            return []
        
        mappings = []
        for i, item in enumerate(data):
            try:
                mapping = PerspectiveMapping(
                    perspective_label=item.get('perspective_label', f'Alternative {i+1}'),
                    perspective_text=item.get('perspective_text', alternatives[i] if i < len(alternatives) else ''),
                    epistemological_frame=PerspectiveFrame(item.get('epistemological_frame', 'pragmatist')),
                    ontological_commitments=item.get('ontological_commitments', []),
                    methodological_commitments=item.get('methodological_commitments', []),
                    frame_confidence=float(item.get('frame_confidence', 0.5)),
                )
                mappings.append(mapping)
            except (ValueError, KeyError):
                continue
        
        return mappings
    
    except (json.JSONDecodeError, ValueError):
        return []


def tag_ontology(
    question: str,
    alternatives: list[str],
    context: Optional[str] = None,
    llm_fn: Optional[Any] = None,
) -> list[OntologyTag]:
    """
    Tag which entities/categories are presupposed by the question and alternatives.
    
    Returns: List of OntologyTag objects
    """
    
    if not llm_fn:
        return []
    
    prompt = f"""Identify entities and categories presupposed by this discussion.

Question: {question}

Alternatives:
{chr(10).join(f'{i+1}. {alt}' for i, alt in enumerate(alternatives))}

Context: {context or 'None'}

For each entity/category that is PRESUPPOSED (i.e., its existence is assumed):
1. Entity name (e.g., "uncertainty", "confidence", "truth")
2. Category (e.g., "epistemic_property", "mental_state", "abstract_object")
3. Which alternatives presuppose it? (list by number: [1, 2, 3])
4. Is it contested? (Do alternatives disagree about whether it exists?)

Return JSON array:
[
  {{
    "entity": "epistemic uncertainty",
    "category": "epistemic_property",
    "presupposed_by": ["1", "2", "3"],
    "is_contested": false
  }},
  ...
]

Return ONLY valid JSON array.
"""
    
    system = "You are an ontological analyst specializing in presupposition detection."
    
    response = llm_fn(system, prompt)
    if not response:
        return []
    
    try:
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            return []
        
        tags = []
        for item in data:
            try:
                tag = OntologyTag(
                    entity=item.get('entity', 'Unknown'),
                    category=item.get('category', 'unknown'),
                    presupposed_by=item.get('presupposed_by', []),
                    is_contested=bool(item.get('is_contested', False)),
                )
                tags.append(tag)
            except (ValueError, KeyError):
                continue
        
        return tags
    
    except (json.JSONDecodeError, ValueError):
        return []


def extract_all(
    question: str,
    alternatives: list[str],
    context: Optional[str] = None,
    llm_fn: Optional[Any] = None,
) -> dict:
    """
    Run full assumption extraction pipeline.
    
    Returns: {
        'latent_assumptions': list[LatentAssumption],
        'perspective_mappings': list[PerspectiveMapping],
        'ontology_tags': list[OntologyTag],
    }
    """
    
    return {
        'latent_assumptions': extract_latent_assumptions(question, alternatives, context, llm_fn),
        'perspective_mappings': map_perspectives(alternatives, context, llm_fn),
        'ontology_tags': tag_ontology(question, alternatives, context, llm_fn),
    }
