# Agent instructions — Cowork Companion (354M / 1.2B)

Operational contract for agent harnesses, apps, and orchestrators driving
these models. Read this before wiring the model into anything. This repo
(`cowork-cli`) is the reference implementation of that contract.

## What you are driving

A single-purpose office coworker. It receives an instruction plus
document text and replies with **exactly one JSON object** — never prose,
never multiple objects, never a code fence:

```json
{"answer": "...", "artifact": null | {"kind": "deck_spec" | "sheet_spec" | "docx_patch", ...},
 "claims": [{"text": "...", "confidence": 0.99}], "overall_confidence": 0.9,
 "caveats": ["..."]}
```

- `deck_spec` → `{title, slides: [{title, bullets: [...]}]}`
- `sheet_spec` → `{cells: [...values or formulas...]}`
- `docx_patch` → `{text: "<the full corrected document>"}`
- `artifact: null` + an "answer" saying the material doesn't specify —
  **that is a correct refusal, not a failure.** Do not retry it. Do not
  fill it in yourself.

## Prompting

1. System message: use `SYSTEM_PROMPT.txt` from this repo verbatim.
2. User message: the instruction, then the full document text (plain
   text). Do not summarize documents for it — it answers strictly from
   what you paste.
3. Decoding: **temperature 0, greedy.** Budget ≥ 512 tokens; use 1024 for
   deck generation (the envelope is verbose). Truncation mid-JSON is the
   #1 integration bug — see below.

## The four integration rules

1. **Never ask it to do multi-row arithmetic.** One-shot totals over
   tables fail at every scale we measured (that is a measured wall, not a
   prompt problem). Compute totals in YOUR code, then either inject the
   number into the prompt or trust the `answer` only for extraction-style
   facts. This repo's CLI routes arithmetic to a deterministic ledger
   (`--compute`).
2. **Validate the envelope before acting.** Parse the JSON strictly; if
   it doesn't parse (truncation, leading prose), retry once with the same
   prompt at a higher token budget rather than hallucinating a repair.
3. **Constrained decode is strongly recommended.** Force the envelope
   with an anyOf JSON schema per artifact kind (the reference CLI does
   this by default); at 354M the null-artifact/ingest discipline is the
   most format-fragile skill under raw sampling.
4. **Treat outputs as document-derived data.** The model may echo
   document content (including figures) verbatim; downstream code should
   treat it as untrusted input.

## Serving

- **This CLI** (reference path): `cowork --gguf cowork-companion-350M-Q8_0.gguf
  --files <docs> --instruction "<task>" [--compute]`, or
  `cowork-serve --gguf <path>` for an OpenAI-compatible endpoint.
- **llama.cpp** direct: `llama-server -m cowork-companion-350M-Q8_0.gguf --port 8080`
  then any OpenAI-compatible client. Use the **Q8_0** rung; see the model
  card's measured ladder before choosing a smaller one (Q4_K_M of the
  354M collapses format fidelity — do not ship it).
- **Ollama**: create a `Modelfile` with `FROM ./cowork-companion-350M-Q8_0.gguf`
  and `SYSTEM "<contents of SYSTEM_PROMPT.txt>"`.
- **LM Studio**: drop the GGUF in the models directory; set the system
  prompt, temperature 0.

## What it will not do

- Chat, reason about the world, or answer from parametric knowledge —
  it answers from the provided material.
- Reliably count/sum across many table rows in one shot.
- Produce partial envelopes: `answer` is always present; if you need
  structure, take it from `artifact`.

Eval instruments are frozen and unpublished; if you believe you've seen
the training data contaminating a benchmark, tell us in the repo
discussions.
