"""Render cowork artifact specs to real Office files.

The bridge between the model's JSON envelope and deliverables:
  deck_spec   -> .pptx   (title slide + content slides with bullets)
  sheet_spec  -> .xlsx   (optionally over a base CSV, applying formulas/values)
  docx_patch  -> .docx   (the corrected text as paragraphs)

Lazy imports (matching renderers.py house style); callers surface
ImportError as a clean "install the office extras" message.
"""
from __future__ import annotations

import csv
import io


def _coerce(v: str):
    """CSV cells arrive as strings; restore numbers so formulas bind."""
    s = v.strip()
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return v


def render_deck_spec(spec: dict, path: str) -> int:
    """Write a deck_spec as .pptx; returns the number of slides written."""
    from pptx import Presentation
    from pptx.util import Pt

    prs = Presentation()
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = str(spec.get("title", "Untitled"))
    slides = spec.get("slides") or []
    for slide_spec in slides:
        layout = prs.slide_layouts[1]  # title + content
        slide = prs.slides.add_slide(layout)
        slide.shapes.title.text = str(slide_spec.get("title", ""))
        body = slide.placeholders[1].text_frame
        bullets = [str(b) for b in (slide_spec.get("bullets") or [])]
        if bullets:
            body.text = bullets[0]
            for b in bullets[1:]:
                para = body.add_paragraph()
                para.text = b
                para.level = 0
        for para in body.paragraphs:
            for run in para.runs:
                run.font.size = Pt(18)
    prs.save(path)
    return len(slides)


def render_sheet_spec(spec: dict, path: str, base_csv: str | None = None) -> int:
    """Write a sheet_spec as .xlsx, optionally applied over a base CSV table.

    With ``base_csv``, the table is written starting at A1 (header row 1)
    and the spec's cells are then applied at their refs (formula strings
    starting with '=' are written as formulas, bare values as values).
    Returns the number of spec cells applied.
    """
    from openpyxl import Workbook
    from openpyxl.utils import column_index_from_string

    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    if base_csv:
        for row in csv.reader(io.StringIO(base_csv.strip())):
            if row:
                ws.append([_coerce(v) for v in row])
    for cell in spec.get("cells") or []:
        ref = str(cell.get("ref", "")).replace("$", "")
        if not ref:
            continue
        col_letters = "".join(ch for ch in ref if ch.isalpha()).upper()
        row_num = int("".join(ch for ch in ref if ch.isdigit()))
        target = ws.cell(row=row_num,
                         column=column_index_from_string(col_letters))
        value = cell.get("value")
        formula = cell.get("formula")
        if formula:
            target.value = str(formula)
        elif value is not None:
            try:
                target.value = float(value) if "." in str(value) else int(value)
            except (TypeError, ValueError):
                target.value = value
    wb.save(path)
    return len(spec.get("cells") or [])


def render_docx_patch(text: str, path: str) -> int:
    """Write corrected text as a .docx; returns the paragraph count."""
    from docx import Document

    doc = Document()
    paras = [p for p in text.split("\n\n") if p.strip()]
    for para in paras:
        doc.add_paragraph(para.strip())
    doc.save(path)
    return len(paras)
