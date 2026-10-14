"""Office-file text extraction for reference injection.

Promoted from ``scripts/verify_claims.py`` so both the eval runners and the
cowork CLI share one implementation. Every extractor degrades to "" when its
library is missing; ``extract_reference`` additionally handles http(s) URLs
with a disk cache and a hard character cap (large real documents must be
truncated or retrieved over - never silently dropped).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse

try:  # docx
    from docx import Document as _Docx
    DOCX_OK = True
except ImportError:  # pragma: no cover
    DOCX_OK = False
try:  # xlsx
    from openpyxl import load_workbook as _XlsxLoad
    XLSX_OK = True
except ImportError:  # pragma: no cover
    XLSX_OK = False
try:  # pptx
    from pptx import Presentation as _Pptx
    PPTX_OK = True
except ImportError:  # pragma: no cover
    PPTX_OK = False
try:  # pdf
    from pypdf import PdfReader as _PdfReader
    PDF_OK = True
except ImportError:  # pragma: no cover
    PDF_OK = False

DEFAULT_MAX_CHARS = 24_000  # ~6k tokens; fits the seq budget with the question


def text_from_docx(path: Path) -> str:
    if not DOCX_OK:
        return ""
    doc = _Docx(str(path))
    paras = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                paras.append(cell.text)
    return "\n".join(paras)


def text_from_xlsx(path: Path) -> str:
    if not XLSX_OK:
        return ""
    wb = _XlsxLoad(str(path), data_only=True, read_only=True)
    # Formula cells carry no cached value until Excel opens the file; a
    # values-only pass makes them invisible. Overlay a formula pass so
    # "=SUM(...)" stays in the text (cached value appended when present).
    try:
        wbf = _XlsxLoad(str(path), data_only=False, read_only=True)
    except Exception:  # noqa: BLE001 - formula pass is best-effort
        wbf = None
    lines = []
    for s_i, sheet in enumerate(wb.worksheets):
        lines.append(f"--- Sheet: {sheet.title} ---")
        f_sheet = (wbf.worksheets[s_i]
                   if wbf is not None and s_i < len(wbf.worksheets) else None)
        f_rows = (list(f_sheet.iter_rows(values_only=True))
                  if f_sheet is not None else [])
        for r_i, row in enumerate(sheet.iter_rows(values_only=True)):
            f_row = f_rows[r_i] if r_i < len(f_rows) else ()
            cells = []
            for c_i, v in enumerate(row):
                fv = f_row[c_i] if c_i < len(f_row) else None
                if isinstance(fv, str) and fv.startswith("="):
                    cells.append(f"{fv} (={v})" if v is not None else str(fv))
                else:
                    # Empty cells are KEPT as empty fields: dropping them
                    # shifts every later column, which breaks formula refs
                    # that point at real cell coordinates.
                    cells.append("" if v is None else str(v))
            if any(c.strip() for c in cells):
                lines.append(" | ".join(cells).rstrip(" |"))
    return "\n".join(lines)


def text_from_pptx(path: Path) -> str:
    if not PPTX_OK:
        return ""
    prs = _Pptx(str(path))
    lines = []
    for i, slide in enumerate(prs.slides):
        lines.append(f"--- Slide {i + 1} ---")
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                lines.append(shape.text)
            if shape.has_table:
                for row in shape.table.rows:
                    lines.append(" | ".join(c.text for c in row.cells))
    return "\n".join(lines)


def text_from_pdf(path: Path, max_pages: int = 40) -> str:
    if not PDF_OK:
        return ""
    reader = _PdfReader(str(path))
    pages = reader.pages[:max_pages]
    return "\n".join((p.extract_text() or "") for p in pages)


def pdf_pages(path: Path, max_pages: int = 200) -> list[str]:
    """Per-page text of a PDF, page-indexed from 0. Degrades to [] when
    pypdf is missing or the file fails - callers count failures."""
    if not PDF_OK:
        return []
    try:
        reader = _PdfReader(str(path))
        return [(p.extract_text() or "") for p in reader.pages[:max_pages]]
    except Exception:  # noqa: BLE001 - one bad file must not kill a run
        return []


_QUESTION_STOP = frozenset(
    "a an and are as at be by for from has have how in is it its of on or "
    "that the to was were what when where which who why will with does do "
    "did can could should would about into over under per page".split())


def _question_tokens(question: str) -> list[str]:
    """Content tokens of a question: lowercase alphanumerics, stopwords and
    1-char tokens removed. Numbers are KEPT (high-signal in doc QA)."""
    import re
    toks = re.findall(r"[a-z0-9]+", question.casefold())
    return [t for t in toks if len(t) >= 2 and t not in _QUESTION_STOP]


def select_pages(pages: list[str], question: str, top_pages: int = 6,
                 ) -> list[int]:
    """Keyword-overlap page ranking for long-PDF QA.

    Scores each page by the number of unique question content tokens it
    contains (tf-capped at 3 per token), returns indices of the best pages
    in PAGE ORDER. Pages with zero overlap are ignored; if every page scores
    zero the first ``top_pages`` pages are returned (the pre-retrieval
    behavior, so the policy only ever adds signal).
    """
    toks = set(_question_tokens(question))
    if not toks or not pages:
        return list(range(min(top_pages, len(pages))))

    def score(page: str) -> float:
        low = page.casefold()
        s = 0.0
        for t in toks:
            c = low.count(t)
            if c:
                s += min(c, 3)
        return s

    ranked = sorted(((score(p), i) for i, p in enumerate(pages)),
                    key=lambda x: (-x[0], x[1]))
    hits = [i for s, i in ranked if s > 0][:top_pages]
    if not hits:
        return list(range(min(top_pages, len(pages))))
    return sorted(hits)


def extract_reference_pages(ref: str, question: str,
                            cache_dir: Path | None = None,
                            top_pages: int = 6,
                            max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Page-retrieval extraction for long PDFs: fetch (cached), rank pages
    against the question, inject only the top pages in page order, each
    labeled 'Page N+1' so the model can cite evidence pages (the XL-Doc
    scorer credits citations). Degrades to "" on failure like
    ``extract_reference``."""
    try:
        if ref.startswith(("http://", "https://")):
            if cache_dir is None:
                return ""
            path = _fetch_url(ref, cache_dir)
        else:
            path = Path(ref)
        if path.suffix.lower() != ".pdf" or not path.exists():
            return ""
        pages = pdf_pages(path)
        if not pages:
            return ""
        keep = select_pages(pages, question, top_pages=top_pages)
        blocks = [f"--- Page {i + 1} ---\n{pages[i]}" for i in keep]
        text = "\n".join(blocks)
    except Exception:  # noqa: BLE001 - one bad reference must not kill a run
        return ""
    return text[:max_chars]


def text_from_plaintext(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


_EXTRACTORS = {
    ".docx": text_from_docx,
    ".xlsx": text_from_xlsx,
    ".xlsm": text_from_xlsx,
    ".pptx": text_from_pptx,
    ".pdf": text_from_pdf,
    ".csv": text_from_plaintext,
    ".md": text_from_plaintext,
    ".txt": text_from_plaintext,
}

SUPPORTED = sorted(_EXTRACTORS)


def _url_cache_path(url: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    suffix = Path(urlparse(url).path).suffix or ".bin"
    return cache_dir / f"extdoc-{digest}{suffix}"


def _fetch_url(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = _url_cache_path(url, cache_dir)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    # Some hosts (e.g. digitallibrary.un.org) stub out default-python-urllib
    # requests with an empty body; send a browser UA and validate the payload.
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    data = b""
    try:
        from urllib.request import Request
        from urllib.request import urlopen as _open
        with _open(Request(url, headers=headers), timeout=120) as resp:
            data = resp.read()
    except Exception:  # noqa: BLE001 - fall through to the empty check
        data = b""
    if dest.exists():
        dest.unlink()
    dest.write_bytes(data)
    return dest


def extract_reference(ref: str, cache_dir: Path | None = None,
                      max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Extract text from one reference (local path or http(s) URL).

    Returns "" for unsupported/missing/failed references - callers should
    count failures, not assume content.
    """
    try:
        if ref.startswith(("http://", "https://")):
            if cache_dir is None:
                return ""
            path = _fetch_url(ref, cache_dir)
        else:
            path = Path(ref)
        ext = path.suffix.lower()
        fn = _EXTRACTORS.get(ext)
        if fn is None or not path.exists():
            return ""
        text = fn(path)
    except Exception:  # noqa: BLE001 - one bad reference must not kill a run
        return ""
    return text[:max_chars]
