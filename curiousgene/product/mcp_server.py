"""cowork-mcp: expose the cowork companion as an MCP server.

Any MCP client (ZCode, Claude Desktop, other agents) can call:

    cowork_run(files, instruction, compute) -> envelope + artifact paths
    cowork_status()                          -> ship artifact + notes

Transport: stdio (the standard for local MCP servers). Register in a client
config as:

    {"command": "python", "args": ["-m", "curiousgene.product.mcp_server"]}

Model selection: env COWORK_GGUF (path to a merged GGUF, e.g. the Q8_0
rung from https://huggingface.co/tablinum/cowork-companion). Required.
COWORK_DEVICE=vulkan enables GPU offload (default).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from curiousgene.product.cowork_cli import (  # noqa: E402
    build_prompt,
    generate_gguf,
    write_artifacts,
)

mcp = FastMCP("cowork")


def _gguf() -> str:
    gguf = os.environ.get("COWORK_GGUF")
    if not gguf:
        raise RuntimeError(
            "set COWORK_GGUF to a merged GGUF path, e.g. "
            "cowork-companion-350M-Q8_0.gguf downloaded from "
            "https://huggingface.co/tablinum/cowork-companion")
    return gguf


def _vulkan() -> bool:
    return os.environ.get("COWORK_DEVICE", "vulkan").lower() == "vulkan"


def _run_envelope(instruction: str, files: list[str], compute: bool,
                  max_new: int) -> dict:
    """Shared pipeline: prompt -> constrained generation -> envelope +
    rendered artifacts -> computed values."""
    from curiousgene.product.formulas import build_grid, evaluate_spec

    prompt, n_fail = build_prompt(instruction, files, 48_000, compute)
    raw = generate_gguf(_gguf(), prompt, max_new, _vulkan(), 8192, True)
    envelope = None
    start = raw.find("{")
    if start != -1:
        try:
            envelope, _ = json.JSONDecoder().raw_decode(raw[start:])
        except json.JSONDecodeError:
            pass
    artifact_dir = ""
    written: list[str] = []
    computed: dict[str, float | None] = {}
    if envelope is not None:
        out_dir = Path(os.environ.get("COWORK_OUT_DIR",
                                      "runs/cowork_mcp"))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "envelope.json").write_text(
            json.dumps(envelope, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        base_csv = None
        for f in files:
            if f.lower().endswith(".csv"):
                base_csv = Path(f).read_text(encoding="utf-8",
                                             errors="replace")
                break
        written = write_artifacts(envelope, out_dir, base_csv)
        artifact_dir = str(out_dir)
        art = envelope.get("artifact")
        if compute and isinstance(art, dict) \
                and art.get("kind") == "sheet_spec":
            from curiousgene.eval.office_text import extract_reference as _er
            grid: dict[str, float] = {}
            for f in files:
                if Path(f).suffix.lower() in (".xlsx", ".xlsm", ".csv"):
                    text = _er(f, max_chars=48_000)
                    if text:
                        grid.update(build_grid(text))
            if grid:
                computed = evaluate_spec(art.get("cells", []), grid)
    return {
        "envelope": envelope,
        "unparsable_raw": None if envelope is not None else raw[:2000],
        "unreadable_files": n_fail,
        "artifact_dir": artifact_dir,
        "artifacts_written": written,
        "computed_values": computed,
    }


@mcp.tool()
def cowork_run(files: list[str], instruction: str,
               compute: bool = False, max_new: int = 1800) -> str:
    """Run the office-coworker agent on real files.

    Args:
        files: paths to .docx/.xlsx/.pptx/.csv files (may be empty).
        instruction: what to do, e.g. "Update the seat count from 240 to
            310" or "What is the travel budget for the rollout?".
        compute: append harness-computed arithmetic (sums/products) for
            spreadsheet questions - exact totals even though the model
            cannot count.
        max_new: generation cap.

    Returns JSON: {envelope, artifacts_written, computed_values,
    artifact_dir}. envelope.artifact is null for pure questions; otherwise
    kind deck_spec/sheet_spec/docx_patch and files are rendered to disk.
    """
    result = _run_envelope(instruction, files, compute, max_new)
    return json.dumps(result, ensure_ascii=False, indent=2, default=str)


@mcp.tool()
def cowork_status() -> str:
    """Report the shipped model, device mode, and known limits."""
    gguf = _gguf()
    return json.dumps({
        "model": gguf,
        "model_present": Path(gguf).exists(),
        "device": "vulkan" if _vulkan() else "cpu",
        "constrained_decode": True,
        "good_at": ["deck outlines", "sheet formulas from CSV tables",
                    "minimal document edits", "document QA",
                    "grounded refusals"],
        "limits": ["multi-row arithmetic beyond ~3 rows (use compute=True)",
                   "out-of-distribution real files can misroute"],
    }, indent=2)


def main() -> int:
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
