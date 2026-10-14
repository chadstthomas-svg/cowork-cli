"""Harness arithmetic — the deterministic computed ledger (`--compute`).

Vendored from the CuriousGene research repo's ``scripts/external_bench_eval.py``
(``_computed_ledger``), byte-for-byte in logic. The measured design rule:
small models cannot do multi-row arithmetic (354M: 3 rows 1.000, 7 rows
0.167; the wall persists at 1.2B), so the harness computes and the model
reads. Python does the arithmetic; the model does reading and composition.
"""
from __future__ import annotations


def computed_ledger(sheet_text: str) -> str:
    """Deterministic arithmetic augmentation for a CSV/TSV-ish sheet block.

    For every table-like block (header row + numeric rows), appends a
    computed ledger: per-row products for Item,Qty,Price-style sheets and a
    SUM per numeric column. This is the harness-assisted mode: Python does
    the arithmetic (mirroring the sheet_spec formula engine), the model does
    reading and composition.
    """
    out_lines = []
    for block in sheet_text.split("\n\n"):
        lines = [ln for ln in block.splitlines()
                 if ln and ("|" in ln or "," in ln or "\t" in ln)]
        if len(lines) < 3:
            continue
        def cells(ln: str) -> list[str]:
            return ([c.strip() for c in ln.split("|") if c.strip()]
                    if "|" in ln else
                    [c.strip() for c in ln.split("\t")] if "\t" in ln
                    else [c.strip() for c in ln.split(",")])
        header = cells(lines[0])
        if len(header) < 2:
            continue
        rows = [cells(ln) for ln in lines[1:]]
        rows = [r for r in rows if len(r) == len(header)]
        if len(rows) < 2:
            continue

        def is_num(s: str) -> bool:
            try:
                float(s.replace(",", ""))
                return True
            except ValueError:
                return False

        num_cols = [i for i in range(len(header)) if is_num(rows[0][i])]
        if len(num_cols) < 1:
            continue
        out_lines.append(f"[computed ledger for table '{header[0]}...']")
        # Per-row products when the sheet is Qty x Price shaped.
        if (len(num_cols) >= 2 and len(header) >= 3
                and all(is_num(r[num_cols[0]]) and is_num(r[num_cols[1]])
                        for r in rows)):
            prods = [float(r[num_cols[0]].replace(",", ""))
                     * float(r[num_cols[1]].replace(",", "")) for r in rows]
            for r, p in zip(rows, prods, strict=True):
                out_lines.append(f"  row [{r[0]}]: {_fmt_num(p)}")
            out_lines.append(f"  SUM of row products: {_fmt_num(sum(prods))}")
        for i in num_cols:
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[i].replace(",", "")))
                except ValueError:
                    vals = []  # column turned out non-numeric (dates etc.)
                    break
            if not vals:
                continue
            out_lines.append(f"  SUM {header[i]}: {_fmt_num(sum(vals))}  "
                             f"COUNT {header[i]}: {len(vals)}  "
                             f"MAX {header[i]}: {_fmt_num(max(vals))}  "
                             f"MIN {header[i]}: {_fmt_num(min(vals))}")
        # Group-by ledger: most real questions are FILTERED aggregates
        # ("attendees with RSVP status 'Yes'"), not whole-column sums. For
        # each low-cardinality text column x numeric column, emit per-group
        # sums/counts.
        cat_cols = [i for i in range(len(header)) if i not in num_cols
                    and len({r[i] for r in rows}) <= 12]
        for ci in cat_cols:
            for ni in num_cols:
                if ci == ni:
                    continue
                groups: dict[str, list[float]] = {}
                ok = True
                for r in rows:
                    try:
                        groups.setdefault(r[ci], []).append(
                            float(r[ni].replace(",", "")))
                    except ValueError:
                        ok = False
                        break
                if not ok:
                    continue
                for v in sorted(groups):
                    gv = groups[v]
                    out_lines.append(
                        f"  SUM {header[ni]} WHERE {header[ci]}='{v}': "
                        f"{_fmt_num(sum(gv))} (COUNT {len(gv)})")
    return "\n".join(out_lines)


def _fmt_num(v: float) -> str:
    return f"{v:,.2f}".rstrip("0").rstrip(".") if v == v else "?"
