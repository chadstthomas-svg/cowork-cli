"""Engram sidecar retrieval — keyword baseline + learned projection (Path C).

The keyword path is a behavior-identical port of the Phase-2 probe
implementation (``scripts/engram_phase2_probe.py``): same STOP set,
same question-coverage score, same first-wins tie-break — the measured
hit@1 0.967 (29/30) transfers unchanged. The projection path (Path C,
``docs/PATH_C_REGISTRATION_2026-10-08.md``) learns one shared
content-word projection scored by cosine similarity; without trained
weights it silently falls back to the keyword result (gate C3).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

try:  # torch is optional in the public distribution: only the learned
    # ProjectionRetriever path needs it; KeywordRetriever is pure Python.
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

STOP = {"the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "in",
        "on", "at", "to", "of", "for", "with", "my", "our", "i", "you", "how",
        "what", "where", "when", "can", "should", "it", "that", "this", "set",
        "right", "correct", "question", "policy", "contoso"}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in STOP and len(w) > 2}


def load_fact_base(path: Path | str | None = None) -> list[dict]:
    """Load a fact base jsonl ({"id", "fact", "for"} per line).

    No default here: point it at your own fact table, or the packaged
    example (``curiousgene.data.EXAMPLE_FACT_BASE``).
    """
    if path is None:
        raise ValueError("pass a fact base path (see "
                         "curiousgene.data.EXAMPLE_FACT_BASE for an example)")
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class KeywordRetriever:
    """Content-word-overlap top-1 (question coverage), first-wins ties.

    Behavior-identical to the Phase-2 probe's ``retrieve()``.
    """

    def __init__(self, fact_base: list[dict]):
        self.base = fact_base

    def retrieve(self, question: str) -> dict:
        qw = words(question)
        best, best_score = None, -1.0
        for f in self.base:
            fw = words(f["fact"])
            score = len(qw & fw) / (len(qw) + 1e-9)  # question coverage
            if score > best_score or (score == best_score and best is None):
                best, best_score = f, score
        return {"fact": best["fact"], "fact_id": best["id"], "for": best["for"],
                "score": round(best_score, 4)}


def keyword_retrieve(question: str, fact_base: list[dict]) -> dict:
    """Module-level parity wrapper over the probe's signature."""
    return KeywordRetriever(fact_base).retrieve(question)


class ProjectionRetriever:
    """Learned retrieval projection (Path C).

    Question and facts are embedded as multi-hot content-word bags over
    the fact-base vocabulary, projected by one shared matrix ``W``
    (V x d), and scored by cosine similarity. Untrained (or failed to
    load) ``W`` degrades to the keyword retriever, never raises.
    """

    def __init__(self, fact_base: list[dict], d: int = 64,
                 weights_path: Path | str | None = None):
        if torch is None:
            raise ImportError("ProjectionRetriever requires torch "
                              "(pip install 'cowork-cli[engram]')")
        self.base = fact_base
        self.keyword = KeywordRetriever(fact_base)
        self.vocab = sorted({w for f in fact_base for w in words(f["fact"])})
        self.index = {w: i for i, w in enumerate(self.vocab)}
        self.d = d
        self.W: torch.Tensor | None = None
        if weights_path is not None:
            self.load(weights_path)

    # -- features ---------------------------------------------------------

    def bag(self, text: str) -> torch.Tensor:
        vec = torch.zeros(len(self.vocab), dtype=torch.float32)
        for w in words(text):
            i = self.index.get(w)
            if i is not None:
                vec[i] = 1.0
        return vec

    def fact_bags(self) -> torch.Tensor:
        return torch.stack([self.bag(f["fact"]) for f in self.base])

    def _fact_vecs(self) -> torch.Tensor:
        assert self.W is not None
        return F.normalize(self.fact_bags() @ self.W, dim=-1)

    # -- retrieval --------------------------------------------------------

    def retrieve(self, question: str) -> dict:
        if self.W is None:
            return self.keyword.retrieve(question)
        q = F.normalize(self.bag(question) @ self.W, dim=-1)
        sims = self._fact_vecs() @ q
        i = int(torch.argmax(sims))
        return {"fact": self.base[i]["fact"], "fact_id": self.base[i]["id"],
                "for": self.base[i]["for"], "score": round(float(sims[i]), 4),
                "method": "projection"}

    # -- training ---------------------------------------------------------

    def fit(self, questions: list[tuple[str, int]], *, epochs: int = 400,
            lr: float = 5e-3, tau: float = 0.07, seed: int = 0,
            verbose: bool = False) -> list[float]:
        """Fit ``W`` by cross-entropy over the fact base.

        ``questions`` is a list of ``(question_text, gold_fact_index)``.
        The 30 real bench questions must never appear here (registration
        discipline); callers own that split.
        """
        g = torch.Generator().manual_seed(seed)
        W = (torch.randn(len(self.vocab), self.d, generator=g) * 0.05
             ).requires_grad_(True)
        q_bags = torch.stack([self.bag(q) for q, _ in questions])
        gold = torch.tensor([i for _, i in questions], dtype=torch.long)
        fact_bags = self.fact_bags()
        opt = torch.optim.AdamW([W], lr=lr)
        history: list[float] = []
        for _ in range(epochs):
            opt.zero_grad()
            qv = F.normalize(q_bags @ W, dim=-1)
            fv = F.normalize(fact_bags @ W, dim=-1)
            loss = F.cross_entropy((qv @ fv.T) / tau, gold)
            loss.backward()
            opt.step()
            history.append(float(loss))
            if verbose and len(history) % 50 == 0:
                print(f"[fit] epoch {len(history)} loss {loss.item():.4f}")
        self.W = W.detach()
        return history

    # -- persistence ------------------------------------------------------

    def save(self, path: Path | str) -> None:
        assert self.W is not None, "nothing trained to save"
        Path(path).write_text(json.dumps({
            "d": self.d,
            "vocab": self.vocab,
            "fact_ids": [f["id"] for f in self.base],
            "W": [[round(float(x), 5) for x in row] for row in self.W],
        }), encoding="utf-8")

    def load(self, path: Path | str) -> None:
        """Gate C3: any load failure leaves the keyword fallback active."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            vocab = data["vocab"]
            if vocab != self.vocab:
                raise ValueError("vocab mismatch with the loaded fact base")
            W = torch.tensor(data["W"], dtype=torch.float32)
            if tuple(W.shape) != (len(vocab), int(data["d"])):
                raise ValueError(f"weight shape {tuple(W.shape)} unexpected")
            self.W = W
        except Exception:
            self.W = None
