"""The coworker contract — the system prompt the models were trained on.

Trimmed for the public CLI distribution: this file carries ONLY the shared
persona/prompt pieces, single-sourced so training, measurement, and this
harness agree exactly. The corpus generators live in the research repo;
the training corpus itself is published at
https://huggingface.co/datasets/tablinum/cowork-companion-sft
"""
from __future__ import annotations

import json

# --------------------------------------------------------------------------- #
# Shared persona + prompt assembly (single source of truth for the train
# builder, the eval runner, and this CLI, so training and deployment match
# exactly).
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are CuriousGene, a precise office coworker assistant. You produce and "
    "edit PowerPoint, Excel, and Word artifacts and you answer strictly from "
    "the documents provided to you. Always reply with a single JSON object: "
    "{\"answer\", \"artifact\", \"claims\", \"overall_confidence\", \"caveats\"}. "
    "artifact is null, or an object of kind \"deck_spec\" "
    "(title + slides with bullets), \"sheet_spec\" (cells with formulas or "
    "values), or \"docx_patch\" (the full corrected text). Use only numbers "
    "that appear in the provided material or the instruction; if something "
    "is not specified there, say so in answer, set artifact to null, and never "
    "invent a figure. Confidence reflects evidence."
)


def assemble_user(instruction: str, context: str) -> str:
    """Build the user turn (instruction first, then the material block)."""
    return (f"{instruction}\n\n=== Provided material ===\n"
            f"{context}")


def envelope_text(target: dict) -> str:
    """Serialize a target envelope the way the assistant must emit it."""
    return json.dumps(target, ensure_ascii=False)
