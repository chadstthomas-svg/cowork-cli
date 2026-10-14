"""cowork - the office companion CLI.

Hand it real files and an instruction; it builds the context from the files
(`curiousgene.eval.office_text`), runs the coworker model, prints the JSON
envelope, and renders artifacts:

    cowork --gguf cowork-companion-350M-Q8_0.gguf --vulkan \
        --files brief.docx --instruction "Update the seat count to 310."

Console entry point: ``cowork`` (pyproject [project.scripts]); the module
also runs directly as ``python -m curiousgene.product.cowork_cli``.

Two backends:
  gguf  - llama-server subprocess (deployment shape, the default; CPU or
          Vulkan via --vulkan) — download a rung from
          https://huggingface.co/tablinum/cowork-companion
  hf    - transformers + optional LoRA adapter (dev/measurement; lives in
          the CuriousGene research repo, not this distribution)

Artifacts from the envelope are written to --out-dir:
  deck_spec -> cowork_deck.pptx, sheet_spec -> cowork_sheet.xlsx,
  docx_patch -> cowork_doc.docx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from curiousgene.data.cowork_tasks import SYSTEM_PROMPT
from curiousgene.eval.office_text import extract_reference
from curiousgene.product.artifacts import (
    render_deck_spec,
    render_docx_patch,
    render_sheet_spec,
)
from curiousgene.product.ledger import computed_ledger

# Material block mirrors the eval-time assemble_user shape (instruction
# first, then "=== Provided material ===") - the product path measured 0.75
# on ingest with the old material-first order vs 1.000 on the eval path.
PREAMBLE = (
    "{instruction}\n\n=== Provided material ===\n{context}\n"
)


# Deterministic file-type guidance appended to the user turn. The model is
# trained on clean single-table CSVs and plain memos; real files (formula
# sheets, numbered-heading documents, existing decks) sometimes route it to
# the wrong artifact kind. The nudge names the right kind without touching
# weights. Scoped so question-style instructions still get artifact null.
_EXT_NUDGES: dict[str, str] = {
    ".xlsx": ("the material includes a spreadsheet: if the instruction asks "
              "to modify, extend, or compute over it, reply with a "
              "sheet_spec artifact listing only the new cells (with "
              "spreadsheet formulas); if it only asks a question, set "
              "artifact to null and answer from the numbers shown"),
    ".xlsm": None,  # resolved to the .xlsx text below
    ".csv": None,
    ".docx": ("the material includes a document: if the instruction asks to "
              "edit it, reply with a docx_patch artifact containing the "
              "FULL corrected text from start to end, changing nothing "
              "beyond the requested edits; a question is answered with "
              "artifact null"),
    ".pptx": ("the material includes a presentation: if the instruction "
              "asks to add or extend slides, reply with a deck_spec "
              "artifact containing the requested new slides (each with "
              "bullets citing the given figures); a question is answered "
              "with artifact null"),
}
_EXT_NUDGES[".xlsm"] = _EXT_NUDGES[".xlsx"]
_EXT_NUDGES[".csv"] = _EXT_NUDGES[".xlsx"]


def build_prompt(instruction: str, files: list[str], max_chars: int,
                 compute: bool = False,
                 engram_path: str | None = None) -> tuple[str, int]:
    blocks, failures, nudges = [], [], set()
    ledgers: list[str] = []
    for f in files:
        text = extract_reference(f, max_chars=max_chars)
        if text:
            if compute and Path(f).suffix.lower() in (".xlsx", ".xlsm",
                                                      ".csv"):
                # Harness arithmetic (--compute): column sums, row products
                # and group-by filtered sums computed in Python. The model
                # cannot do multi-row arithmetic itself; the ledger is
                # placed next to the instruction (closest to the generation
                # point) so the numbers are copied, not recomputed.
                try:
                    ledger = computed_ledger(text)
                except Exception:  # noqa: BLE001 - one odd sheet must not
                    ledger = ""    # kill a request
                if ledger:
                    ledgers.append(ledger)
            blocks.append(f"--- {f} ---\n{text}")
            nudge = _EXT_NUDGES.get(Path(f).suffix.lower())
            if nudge:
                nudges.add(nudge)
        else:
            failures.append(f)
    context = "\n\n".join(blocks) if blocks else "(no readable files provided)"
    instruction_text = instruction
    if engram_path:
        # E-VERTICAL (--engram): retrieval-v2 (bge top-10 → cross-encoder
        # re-rank → keyword tiebreak) over the fact base, injected beside
        # the instruction. Falls back to the keyword retriever when the
        # models can't load (cold cache, offline box). Facts ride outside
        # the weights; the retrieval is deterministic and logged to the run.
        import json as _json
        facts = [_json.loads(line) for line in
                 Path(engram_path).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        hit = None
        if facts:
            try:
                from curiousgene.engram.retrieval_v2 import RetrievalV2
                hit = RetrievalV2(facts).retrieve(instruction)
            except Exception:  # noqa: BLE001 - one cold cache must not
                from curiousgene.engram.retrieval import KeywordRetriever
                hit = KeywordRetriever(facts).retrieve(instruction)
        if hit and instruction.strip():
            instruction_text += (f"\n\nKnown fact: {hit['fact']}")
    if ledgers:
        instruction_text += ("\n\n[Computed totals - arithmetic done by the "
                             "harness; COPY these numbers, do not "
                             "recompute:]\n" + "\n".join(ledgers))
    if nudges:
        instruction_text += ("\n\nResponse format notes: "
                             + "; ".join(sorted(nudges)) + ".")
    return (PREAMBLE.format(context=context, instruction=instruction_text),
            len(failures))


def generate_hf(base: str, adapter: str | None, bf16: bool, prompt: str,
                max_new: int) -> str:
    import torch
    try:
        # load_model lives in scripts/cowork_eval.py (namespace import) so
        # the dev backend and the measurement runner share one
        # implementation — research-repo only.
        from scripts.cowork_eval import load_model
    except ImportError:
        raise SystemExit(
            "the --backend hf path lives in the CuriousGene research "
            "repo; this distribution ships the deployment path "
            "(--backend gguf, the default)") from None
    m, tok = load_model(base, adapter, bf16)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(m.device)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


class LlamaServer:
    """One-shot llama-server context manager (BenchServer pattern: own port,
    own process, always stopped; /health readiness poll). CPU by default,
    Vulkan offload with --vulkan."""

    def __init__(self, gguf: str, ctx: int, vulkan: bool, threads: int = 6):
        self.gguf, self.ctx, self.vulkan, self.threads = gguf, ctx, vulkan, threads
        self.proc = None
        self.port = self._free_port()

    @staticmethod
    def _free_port() -> int:
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def __enter__(self) -> str:
        import shutil
        import subprocess
        import time
        binary = shutil.which("llama-server")
        if not binary:
            raise RuntimeError("llama-server not on PATH")
        log_dir = Path("runs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log = (log_dir / "cowork_cli_server.log").open("a",
                                                       encoding="utf-8")
        cmd = [binary, "--model", self.gguf, "--host", "127.0.0.1",
               "--port", str(self.port), "--ctx-size", str(self.ctx),
               "--parallel", "1", "--threads", str(self.threads)]
        cmd += (["--n-gpu-layers", "99", "--device", "Vulkan1"] if self.vulkan
                else ["--n-gpu-layers", "0"])
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        endpoint = f"http://127.0.0.1:{self.port}"
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                from urllib.request import urlopen
                urlopen(f"{endpoint}/health", timeout=2)
                return endpoint
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        "llama-server exited early; see "
                        "runs/cowork_cli_server.log") from None
                time.sleep(1.0)
        raise RuntimeError("llama-server startup timed out")

    def __exit__(self, *exc) -> None:
        import subprocess
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)


# JSON schema for the house envelope. Passed to llama-server as a decode
# constraint (--constrained): keys, kinds and types are enforced by the
# grammar while all content stays free. This removes the quantized models'
# token-level format corruption (e.g. a "ref" key quantizing to "refresh")
# without touching weights.
ENVELOPE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "artifact": {
            # Per-kind shapes via anyOf: kind is REQUIRED whenever the
            # artifact is present (a bare dict without kind renders nothing).
            "anyOf": [
                {"type": "null"},
                {"type": "object",
                 "properties": {"kind": {"const": "deck_spec"},
                                "title": {"type": "string"},
                                "slides": {"type": "array", "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "bullets": {"type": "array",
                                                    "items": {"type": "string"}},
                                    },
                                    "required": ["title", "bullets"]}}},
                 "required": ["kind", "slides"]},
                {"type": "object",
                 "properties": {"kind": {"const": "sheet_spec"},
                                "cells": {"type": "array", "items": {
                                    "type": "object",
                                    "properties": {
                                        "ref": {"type": "string"},
                                        "formula": {"type": "string"},
                                        "value": {},
                                    },
                                    "required": ["ref"]}}},
                 "required": ["kind", "cells"]},
                {"type": "object",
                 "properties": {"kind": {"const": "docx_patch"},
                                "text": {"type": "string"}},
                 "required": ["kind", "text"]},
            ],
        },
        "claims": {"type": "array", "items": {"type": "object"}},
        "overall_confidence": {"type": "number"},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "artifact", "claims", "overall_confidence",
                 "caveats"],
}


def generate_gguf(gguf: str, prompt: str, max_new: int, vulkan: bool,
                  ctx: int, constrained: bool = False) -> str:
    import json as _json
    from urllib.request import Request, urlopen
    chat_prompt = (
        f"<|startoftext|><|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n")
    with LlamaServer(gguf, ctx, vulkan) as endpoint:
        payload_out = {"prompt": chat_prompt, "n_predict": max_new,
                       "temperature": 0.0}
        if constrained:
            payload_out["json_schema"] = ENVELOPE_JSON_SCHEMA
        body = _json.dumps(payload_out).encode("utf-8")
        req = Request(f"{endpoint}/completion", data=body,
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=600) as resp:
                payload = _json.loads(resp.read().decode("utf-8"))
        except Exception:
            if not constrained:
                raise
            # Older servers reject json_schema - retry unconstrained.
            body = _json.dumps({"prompt": chat_prompt, "n_predict": max_new,
                                "temperature": 0.0}).encode("utf-8")
            req = Request(f"{endpoint}/completion", data=body,
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=600) as resp:
                payload = _json.loads(resp.read().decode("utf-8"))
    return str(payload.get("content", "")).strip()


def write_artifacts(envelope: dict, out_dir: Path, base_csv: str | None) -> list[str]:
    written = []
    art = envelope.get("artifact")
    if not isinstance(art, dict):
        return written
    kind = art.get("kind")
    try:
        if kind == "deck_spec":
            n = render_deck_spec(art, str(out_dir / "cowork_deck.pptx"))
            written.append(f"cowork_deck.pptx ({n} slides)")
        elif kind == "sheet_spec":
            n = render_sheet_spec(art, str(out_dir / "cowork_sheet.xlsx"),
                                  base_csv=base_csv)
            written.append(f"cowork_sheet.xlsx ({n} cells)")
        elif kind == "docx_patch":
            n = render_docx_patch(str(art.get("text", "")),
                                  str(out_dir / "cowork_doc.docx"))
            written.append(f"cowork_doc.docx ({n} paragraphs)")
    except ImportError as exc:
        print(f"[cowork] artifact rendering unavailable: {exc}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--backend", choices=("hf", "gguf"), default="gguf")
    ap.add_argument("--base", default="LiquidAI/LFM2.5-350M-Base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gguf", default=None, help="path to merged GGUF (gguf backend)")
    ap.add_argument("--vulkan", action="store_true", help="GPU-offload llama-server")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-chars", type=int, default=48_000,
                    help="per-file extraction cap (~12k tokens)")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--constrained", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="gguf backend: constrain decoding to the envelope "
                         "JSON schema (removes quant format corruption; "
                         "content stays free). DEFAULT ON; disable with "
                         "--no-constrained")
    ap.add_argument("--compute", action="store_true",
                    help="append a harness-computed arithmetic ledger "
                         "(sums, products, group-bys) to ingested sheets - "
                         "the model reads, Python counts")
    ap.add_argument("--engram", metavar="PATH", nargs="?", const="default",
                    help="inject sidecar-retrieved facts beside the "
                         "instruction (E-VERTICAL; retrieval v2: bge → "
                         "cross-encoder rerank → keyword tiebreak, keyword "
                         "fallback; optional value = your fact base jsonl; "
                         "default = the packaged example fact base)")
    ap.add_argument("--out-dir", default="runs/cowork_cli")
    args = ap.parse_args()

    engram_path = None
    if args.engram:
        if args.engram == "default":
            from curiousgene.data import EXAMPLE_FACT_BASE
            engram_path = str(EXAMPLE_FACT_BASE)
        else:
            engram_path = args.engram
        if not Path(engram_path).exists():
            raise SystemExit(f"engram fact base not found: {engram_path}")

    prompt, n_fail = build_prompt(args.instruction, args.files, args.max_chars,
                                  args.compute, engram_path=engram_path)
    if n_fail:
        print(f"[cowork] warning: {n_fail} file(s) unreadable/unsupported")
    if args.backend == "gguf":
        if not args.gguf:
            print("[cowork] --gguf is required with --backend gguf")
            return 1
        raw = generate_gguf(args.gguf, prompt, args.max_new, args.vulkan,
                            args.ctx, args.constrained)
    else:
        raw = generate_hf(args.base, args.adapter, args.bf16, prompt,
                          args.max_new)

    start = raw.find("{")
    envelope = None
    if start != -1:
        try:
            envelope, _ = json.JSONDecoder().raw_decode(raw[start:])
        except json.JSONDecodeError:
            # Deploy-side repair: quantized models occasionally drop the
            # closing brace(s) of the envelope. Try auto-closing (bounded);
            # eval never uses this path.
            tail = raw[start:].rstrip()
            for extra in ("}", "}}", "}}}", "]","]}"):
                try:
                    candidate, _ = json.JSONDecoder().raw_decode(tail + extra)
                    if isinstance(candidate, dict) and "answer" in candidate:
                        envelope = candidate
                        print("[cowork] note: auto-closed a truncated JSON "
                              "envelope (deploy-side repair)")
                        break
                except json.JSONDecodeError:
                    continue
    if envelope is None:
        print("[cowork] model did not return a JSON envelope; raw output:")
        print(raw)
        if args.compute:
            # Tool-use fallback: even with no usable envelope, hand the
            # user the exact arithmetic from the spreadsheets.
            printed = False
            for f in args.files:
                if Path(f).suffix.lower() not in (".xlsx", ".xlsm", ".csv"):
                    continue
                text = extract_reference(f, max_chars=args.max_chars)
                if not text:
                    continue
                try:
                    ledger = computed_ledger(text)
                except Exception:  # noqa: BLE001
                    ledger = ""
                if ledger:
                    printed = True
                    print(f"[cowork] harness-computed ledger for {f}:")
                    print(ledger)
            if printed:
                return 1
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "envelope.json").write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    base_csv = None
    for f in args.files:
        if f.lower().endswith(".csv"):
            base_csv = Path(f).read_text(encoding="utf-8", errors="replace")
            break
    written = write_artifacts(envelope, out_dir, base_csv)

    print(json.dumps(envelope, ensure_ascii=False, indent=2))
    print(f"[cowork] envelope -> {out_dir / 'envelope.json'}")
    for w in written:
        print(f"[cowork] artifact -> {out_dir / w.split(' ')[0]} ({w})")

    # Tool-use layer (--compute): evaluate the sheet_spec's formulas against
    # a grid harvested from the extracted sheets, and state exact values the
    # model itself cannot compute.
    art = envelope.get("artifact")
    if (args.compute and isinstance(art, dict)
            and art.get("kind") == "sheet_spec" and args.files):
        from curiousgene.product.formulas import build_grid, evaluate_spec, summarize
        grid: dict[str, float] = {}
        for f in args.files:
            if Path(f).suffix.lower() not in (".xlsx", ".xlsm", ".csv"):
                continue
            text = extract_reference(f, max_chars=args.max_chars)
            if text:
                grid.update(build_grid(text))
        if grid:
            values = evaluate_spec(art.get("cells", []), grid)
            if any(v is not None for v in values.values()):
                print("[cowork] computed (harness arithmetic, exact):")
                print(summarize(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
