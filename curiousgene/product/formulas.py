"""Formula evaluation for sheet_spec artifacts (the tool-use layer).

The 350M model cannot do multi-row arithmetic (measured: it fails with the
answers in context). This module is the product's answer: a small, safe
evaluator that computes sheet_spec formulas against a cell grid harvested
from the extracted sheet text, so the CLI can state exact totals even when
the model cannot.

Grid convention: ``build_grid`` maps the extracted sheet text line-by-line
(row 1 = first line, columns A.. = first field), which matches how
llama-visible extractions of real workbooks line up with their formulas
(title, blank line, header, data rows).

Usage:
    grid = build_grid(sheet_text)
    values = evaluate_spec(cells, grid)   # {"D9": 15834.36, ...}
"""
from __future__ import annotations

import re
from decimal import InvalidOperation

_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _col_letters(idx: int) -> str:
    """0-based column index -> spreadsheet letters (0 -> A)."""
    out = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        out = chr(65 + rem) + out
    return out


def _parse_number(text: str) -> float | None:
    t = text.strip().replace(",", "").replace("$", "")
    if not t or not _NUM_RE.match(t):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def build_grid(sheet_text: str) -> dict[str, float]:
    """Extracted sheet text -> {\"A1\": value} for every numeric cell.

    Lines map to rows in order (blank lines still consume a row number,
    matching real-workbook extraction); fields within a line map to
    columns. Formula strings (\"=...\") are skipped - the spec overlays
    them.
    """
    grid: dict[str, float] = {}
    for r, line in enumerate(sheet_text.splitlines(), start=1):
        if "|" in line:
            fields = [f.strip() for f in line.split("|")]
        elif "," in line:
            fields = [f.strip() for f in line.split(",")]
        elif "\t" in line:
            fields = [f.strip() for f in line.split("\t")]
        else:
            continue
        for c, field in enumerate(fields):
            if field.startswith("="):
                # text_from_xlsx overlays cached values as "=F (=v)" - keep
                # them, they are what formulas over raw data need.
                m = re.match(r"^=.*?\(=\s*([^)]+)\)\s*$", field)
                if m:
                    num = _parse_number(m.group(1))
                    if num is not None:
                        grid[f"{_col_letters(c)}{r}"] = num
                continue
            num = _parse_number(field)
            if num is not None:
                grid[f"{_col_letters(c)}{r}"] = num
    return grid


def _range_cells(rng: str) -> list[str]:
    m = re.match(r"^([A-Z]+)(\d+):([A-Z]+)(\d+)$", rng.strip())
    if not m:
        return []
    c1, r1, c2, r2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))

    def col_idx(letters: str) -> int:
        n = 0
        for ch in letters:
            n = n * 26 + (ord(ch) - 64)
        return n - 1

    out = []
    for ci in range(col_idx(c1), col_idx(c2) + 1):
        for ri in range(r1, r2 + 1):
            out.append(f"{_col_letters(ci)}{ri}")
    return out


def _arith(expr: str) -> float:
    """Evaluate a validated arithmetic expression (digits, operators,
    parentheses, dots only) - safe after the caller's regex gate."""
    if not re.match(r"^[\d.+\-*/() ]+$", expr):
        raise ValueError(f"unsafe arithmetic: {expr!r}")
    code = compile(expr, "<formula>", "eval")
    return float(eval(code, {"__builtins__": {}}, {}))  # noqa: S307 - regex-gated


class _Formula:
    def __init__(self, formula: str, grid: dict[str, float],
                 memo: dict[str, float], spec: dict[str, dict],
                 visiting: set[str]) -> None:
        self.f = formula
        self.grid = grid
        self.memo = memo
        self.spec = spec
        self.visiting = visiting

    def cell(self, ref: str) -> float:
        if ref in self.memo:
            return self.memo[ref]
        if ref in self.spec:
            # Chained ref into the spec itself (e.g. =D7+D8): evaluate the
            # referenced spec cell recursively; cycles raise.
            if ref in self.visiting:
                raise ValueError(f"cycle at {ref}")
            cell = self.spec[ref]
            formula = cell.get("formula")
            if formula:
                v = _Formula(str(formula), self.grid, self.memo, self.spec,
                             self.visiting | {ref}).value()
            elif cell.get("value") is not None:
                v = float(cell["value"])
            else:
                raise ValueError(f"no value at {ref}")
            self.memo[ref] = v
            return v
        raw = self.grid.get(ref)
        if raw is None:
            raise ValueError(f"no value at {ref}")
        if isinstance(raw, str) and raw.startswith("="):
            v = _Formula(raw, self.grid, self.memo, self.spec,
                         self.visiting | {ref}).value()
            self.memo[ref] = v
            return v
        return float(raw)

    def value(self) -> float:
        expr = self.f.lstrip("=").replace("^", "**")

        def func(m: re.Match) -> str:
            fn, rng = m.group(1), m.group(2)
            vals = [self.cell(c) for c in _range_cells(rng)]
            table = {"SUM": sum, "AVERAGE": lambda v: sum(v) / len(v),
                     "MIN": min, "MAX": max}
            return repr(table[fn](vals))

        expr = re.sub(
            r"\b(SUM|AVERAGE|MIN|MAX)\(([A-Z]+\d+:[A-Z]+\d+)\)", func, expr,
            flags=re.IGNORECASE)
        expr = re.sub(
            r"\b([A-Z]+\d+)\b",
            lambda m: repr(self.cell(m.group(1))),
            expr)
        return _arith(expr)


def evaluate_spec(cells: list[dict], grid: dict[str, float],
                  ) -> dict[str, float | None]:
    """Evaluate sheet_spec cells against the grid.

    Returns {ref: value} for every cell whose formula (or value) resolves;
    None for cells that cannot (unknown refs, non-numeric sheets). Spec
    cells may reference each other; the grid provides raw data cells.
    """
    spec: dict[str, dict] = {}
    for cell in cells:
        ref = str(cell.get("ref", "")).upper()
        if ref:
            spec[ref] = cell
    memo: dict[str, float] = {}
    out: dict[str, float | None] = {}
    for ref, cell in spec.items():
        formula = cell.get("formula")
        try:
            if formula:
                out[ref] = _Formula(str(formula), grid, memo, spec,
                                    {ref}).value()
            elif cell.get("value") is not None:
                out[ref] = float(cell["value"])
            else:
                out[ref] = None
        except (ValueError, ZeroDivisionError, InvalidOperation, KeyError,
                AttributeError, RecursionError):
            out[ref] = None
    return out


def summarize(values: dict[str, float | None]) -> str:
    """One line per resolvable cell, for the CLI output."""
    rows = []
    for ref in sorted(values):
        v = values[ref]
        rows.append(f"  {ref} = {v:,.2f}" if v is not None
                    else f"  {ref} = (unresolvable)")
    return chr(10).join(rows)
