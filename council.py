#!/usr/bin/env python3
"""
council.py — AI-råd baserat på CognOS-principer

Rådsmedlemmar via Ollama diskuterar en fråga parallellt.
CognOS-divergens mäts som semantisk spridning mellan svaren.
mistral-large-3 syntetiserar med epistemic map.

Användning:
  python3 council.py "Din fråga här"
  python3 council.py "Din fråga" --verbose
"""

import sys
import json
import time
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

OLLAMA_URL = "http://localhost:11434/api/chat"

COUNCIL = [
    {
        "model": "kimi-k2.5:cloud",
        "role": "analytiker",
        "system": (
            "Du är en analytisk tänkare. Undersök frågan noggrant, "
            "identifiera antaganden, och ge ett välgrundat svar. "
            "Var explicit om osäkerhet. Svara kortfattat (max 150 ord)."
        ),
    },
    {
        "model": "llama3.2:1b",
        "role": "skeptiker",
        "system": (
            "Du är en kritisk skeptiker. Utmana antaganden, hitta svagheter "
            "i resonemang, och lyft alternativa förklaringar. "
            "Var direkt. Svara kortfattat (max 150 ord)."
        ),
    },
    {
        "model": "deepseek-r1:1.5b",
        "role": "reasoner",
        "system": (
            "Du är en noggrann resonatör. Tänk steg för steg, visa ditt "
            "resonemang explicit, och var tydlig om var du är osäker. "
            "Svara kortfattat (max 150 ord)."
        ),
    },
]

SYNTHESIZER = "mistral-large-3:675b-cloud"


def ask_model(member: dict, question: str, timeout: int = 60) -> dict:
    """Anropa en rådsmedlem via Ollama API."""
    t0 = time.time()
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": member["model"],
                "messages": [
                    {"role": "system", "content": member["system"]},
                    {"role": "user", "content": question},
                ],
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 200},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["message"]["content"].strip()
        elapsed = round(time.time() - t0, 1)
        return {"model": member["model"], "role": member["role"], "text": text, "elapsed": elapsed, "ok": True}
    except Exception as e:
        return {"model": member["model"], "role": member["role"], "text": "", "elapsed": 0, "ok": False, "error": str(e)}


def jaccard_similarity(a: str, b: str) -> float:
    """Enkelt ordöverlapp som likhetsmått."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def compute_divergence(responses: list[dict]) -> dict:
    """
    CognOS-inspirerad divergensmätning.
    Divergens = 1 - medel-likhet mellan alla par.
    0.0 = alla säger samma sak (suppressed variance)
    1.0 = total oenighet (undirected variance)
    """
    texts = [r["text"] for r in responses if r["ok"] and r["text"]]
    if len(texts) < 2:
        return {"score": 0.0, "regime": "okänt", "pairs": []}

    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = jaccard_similarity(texts[i], texts[j])
            pairs.append(round(1 - sim, 3))  # divergens per par

    score = round(sum(pairs) / len(pairs), 3)

    if score < 0.3:
        regime = "suppressed"      # alla säger samma → osäkerhet dold
    elif score < 0.7:
        regime = "calibrated"      # hälsosam spridning
    else:
        regime = "undirected"      # kaotisk oenighet

    return {"score": score, "regime": regime, "pairs": pairs}


def synthesize(question: str, responses: list[dict], divergence: dict) -> str:
    """phi3:mini syntetiserar rådsrösterna."""
    council_text = "\n\n".join(
        f"[{r['role'].upper()}]\n{r['text']}"
        for r in responses if r["ok"] and r["text"]
    )

    system = (
        "Du är en epistemisk syntetiserare. Du har fått svar från flera "
        "tänkare med olika roller. Din uppgift:\n"
        "1. Identifiera var de är överens\n"
        "2. Identifiera var de är oeniga och VARFÖR\n"
        "3. Ge ett syntetiserat svar som bevarar viktig osäkerhet\n"
        "Var explicit: skriv 'OSÄKERT:' när du inte vet.\n"
        "Max 200 ord."
    )

    prompt = (
        f"Fråga: {question}\n\n"
        f"Rådsröster:\n{council_text}\n\n"
        f"Divergenspoäng: {divergence['score']} ({divergence['regime']})\n\n"
        "Syntetisera:"
    )

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": SYNTHESIZER,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 300},
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(description="Lokalt AI-råd")
    parser.add_argument("question", help="Frågan att diskutera")
    parser.add_argument("--verbose", "-v", action="store_true", help="Visa alla rådsröster")
    args = parser.parse_args()

    question = args.question
    print(f"\n{'='*60}")
    print(f"RÅDET DISKUTERAR: {question}")
    print(f"{'='*60}\n")

    # Parallella anrop
    print("Frågar rådsmedlemmarna parallellt...")
    responses = [None] * len(COUNCIL)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(ask_model, member, question): i for i, member in enumerate(COUNCIL)}
        for future in as_completed(futures):
            i = futures[future]
            responses[i] = future.result()
            r = responses[i]
            status = f"✓ {r['elapsed']}s" if r["ok"] else f"✗ {r.get('error','')}"
            print(f"  [{r['role'].upper()}] ({r['model']}) {status}")

    # Divergensmätning
    divergence = compute_divergence(responses)
    regime_label = {
        "suppressed": "⚠ Supprimerad (alla säger samma)",
        "calibrated": "✓ Kalibrerad (hälsosam spridning)",
        "undirected": "⚠ Okalibrerad (kaotisk oenighet)",
        "okänt": "? Okänt",
    }

    print(f"\nDIVERGENS: {divergence['score']} — {regime_label.get(divergence['regime'], divergence['regime'])}")

    # Visa rådsröster om verbose
    if args.verbose:
        print(f"\n{'─'*60}")
        for r in responses:
            if r["ok"]:
                print(f"\n[{r['role'].upper()}] ({r['model']})")
                print(r["text"])
            else:
                print(f"\n[{r['role'].upper()}] FEL: {r.get('error','')}")

    # Syntes
    ok_responses = [r for r in responses if r["ok"] and r["text"]]
    if not ok_responses:
        print("\nInga svar att syntetisera.")
        return

    print(f"\n{'─'*60}")
    print("SYNTETISERAR...\n")
    synthesis = synthesize(question, ok_responses, divergence)

    print("RÅDETS SLUTSATS:")
    print(f"{'─'*60}")
    print(synthesis)
    print(f"\n{'='*60}")
    print(f"Epistemic regime: {divergence['regime']} | Divergens: {divergence['score']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
