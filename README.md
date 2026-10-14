# cowork — the harness that makes a 354M model a usable office coworker

![license](https://img.shields.io/badge/license-Apache--2.0-gray) ![engine](https://img.shields.io/badge/engine-llama.cpp-8B5CF6) ![org](https://img.shields.io/badge/org-Tablinum-C9A862)

This is the deployment harness for the **Cowork Companion** models
([354M](https://huggingface.co/tablinum/cowork-companion) ·
[1.2B](https://huggingface.co/tablinum/cowork-companion-1.2b), on Hugging Face)
and the public half of a measured design rule:

> **Behavior in the weights, facts in the tables, arithmetic in the harness.**

The models build deck outlines, edit spreadsheets and documents, answer
strictly from provided material, and refuse when the material doesn't answer —
always as exactly one JSON object. What they cannot do at this scale is
multi-row arithmetic (measured: 3-row sums 1.000, 7-row sums 0.167 at 354M;
the wall persists at 1.2B). This CLI is the other half of the product: it
**computes the arithmetic deterministically** (`--compute`), **constrains
decoding to the envelope schema** (default on), **validates the envelope**,
and **renders artifacts to real .pptx/.xlsx/.docx files**.

## Install

```bash
pip install .                    # from a checkout
pip install ".[office]"          # + artifact rendering (.pptx/.xlsx/.docx)
pip install ".[office,engram]"   # + the neural sidecar retriever
```

You also need [llama-server](https://github.com/ggml-org/llama.cpp) on PATH,
and a model rung — **Q8_0 is the measured recommendation**:

```bash
curl -L -o cowork-companion-350M-Q8_0.gguf \
  https://huggingface.co/tablinum/cowork-companion/resolve/main/cowork-companion-350M-Q8_0.gguf
```

## Use

```bash
cowork --gguf cowork-companion-350M-Q8_0.gguf \
       --files budget.csv --compute \
       --instruction "What is the total travel spend for attendees with RSVP status Yes?"

cowork --gguf cowork-companion-350M-Q8_0.gguf \
       --files brief.docx \
       --instruction "Update the seat count from 240 to 310 and return the full corrected memo."
```

Each run prints the JSON envelope (`answer`, `artifact`, `claims`,
`overall_confidence`, `caveats`) and writes rendered artifacts
(`cowork_deck.pptx` / `cowork_sheet.xlsx` / `cowork_doc.docx`) plus
`envelope.json` to `--out-dir` (default `runs/cowork_cli`).

With `--compute`, Python does the arithmetic: column sums, row products and
group-by filtered sums are computed exactly and handed to the model as a
ledger to copy — never to recompute. Even if the model fails entirely, the
CLI still prints the exact totals for your sheets.

### Three entry points

| command | what it does |
|---|---|
| `cowork` | one-shot: files + instruction → envelope + artifacts + computed totals |
| `cowork-serve --gguf <path>` | OpenAI-compatible endpoint (`/v1/chat/completions`) for any client |
| `cowork-mcp` | MCP server (`cowork_run`, `cowork_status`) for agent clients — set `COWORK_GGUF` |

### The Tabula fact sidecar (`--engram`)

A small model will never know your tenant's policies. `--engram <facts.jsonl>`
retrieves from **your** fact table (one `{"id", "fact", "for"}` per line) and
injects the top fact beside the instruction — zero training, measured lift.
Retrieval v2 (bge top-k → cross-encoder re-rank → keyword tiebreak) is used
when the `[engram]` extras are installed; the pure-Python keyword retriever
is the always-available fallback. A tiny example fact base ships in the
package (`curiousgene/data/example_fact_base.jsonl`); pass your own for real
use.

## The operational contract (read before wiring anything)

[AGENTS.md](AGENTS.md) is the contract for harnesses and orchestrators:
envelope validation, the never-do-arithmetic rule, constrained decode, and
refusal semantics (`artifact: null` + "not specified in the material" is a
**correct refusal**, not a failure). [SYSTEM_PROMPT.txt](SYSTEM_PROMPT.txt)
is the system message the models were trained on — use it verbatim.

## Provenance and adaptation notes

Extracted from the CuriousGene research repository, which remains the source
of truth for measurement. Differences from the research copy, all deployment
adaptations with no behavior change on the gguf path:

- the harness ledger is vendored as `curiousgene/product/ledger.py` (logic
  identical to the research repo's eval-side ledger);
- the default backend is `gguf` (the public artifact); the `hf` dev backend
  stays in the research repo and errors with a pointer;
- the default `--engram` table is the packaged example (the registered
  vertical fact tables are measurement artifacts and are not redistributed);
- `torch` is optional — only the learned projection retriever needs it.

The training corpus behind both models is published:
[cowork-companion-sft](https://huggingface.co/datasets/tablinum/cowork-companion-sft)
(3,258 rows, with third-party attribution in its license file). Eval
instruments are deliberately unpublished to keep them uncontaminated;
instrument **hashes** are published on each model repo (`INSTRUMENTS.sha256`).

## License

Code: **Apache-2.0**. The models are separate artifacts under the LFM Open
License v1.0 (see the model cards for the revenue-threshold condition), and
the dataset under the Tablinum Dataset License v1.0.
