"""Retrieval v2 — bge embeddings → cross-encoder re-rank → keyword tiebreak.

Production retriever for the E-VERTICAL engram sidecar (the .9 engineering
plan Phase R/S, ``docs/MASTER_PLAN_2026-10-10.md`` §8g):

1. ``BAAI/bge-small-en-v1.5`` (33M, MIT) embeds the facts and the question
   (batched 128, CLS pooling, L2-normalized; queries carry the model card's
   ``Represent this sentence for searching relevant passages: `` prefix,
   facts carry none) and keeps the top-10 by cosine.
2. ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (22M, Apache-2.0) re-ranks those
   candidates — it reads question+fact jointly, which is what recovers the
   contradiction/polarity misses cosine is blind to. Measured on the full
   18K M365 table: keyword 0.490 → bge 0.755 → bge+rerank 0.898 hit@1
   (``runs/v2full/rerank_results.json``).
3. When the top-2 re-rank scores are within 0.01 the decision is ambiguous
   and the keyword retriever (``KeywordRetriever``, unchanged) breaks the
   tie.

Models resolve from the local HF cache (``HF_HOME``, default ``D:/hf-cache``,
``<HF_HOME>/hub``) via ``local_files_only=True``; a cache-cold model is
downloaded once, and every load — including that same process's later loads —
ends in an offline (local-files-only) resolution. Re-ranker load failure
degrades to bge-only (``method: "bge"``); embedder load failure raises so
callers can fall back to the keyword retriever (the cowork CLI does).
Fact embeddings are computed once, lazily, on the first ``.retrieve()`` and
cached in memory.
"""

from __future__ import annotations

import os

# The hub cache lives outside the repo default; make the documented default
# effective for anything imported after this module (huggingface_hub reads
# HF_HOME at import time, so this must precede those imports).
os.environ.setdefault("HF_HOME", "D:/hf-cache")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import (  # noqa: E402
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from curiousgene.engram.retrieval import KeywordRetriever  # noqa: E402

HF_HOME = os.environ.get("HF_HOME", "D:/hf-cache")
CACHE_DIR = f"{HF_HOME}/hub"


def _resolve(loader, model_id: str):
    """Load ``model_id`` from the local cache; download once if missing.

    Every successful path ends in a ``local_files_only=True`` load, so only
    the very first (cache-cold) call touches the network.
    """
    try:
        return loader.from_pretrained(model_id, cache_dir=CACHE_DIR,
                                      local_files_only=True)
    except OSError:
        snapshot_download(model_id, cache_dir=CACHE_DIR)
        return loader.from_pretrained(model_id, cache_dir=CACHE_DIR,
                                      local_files_only=True)


class RetrievalV2:
    """Three-leg retriever over a fact base of ``{id, fact, for}`` rows.

    ``retrieve()`` returns the KeywordRetriever dict shape
    (``fact``/``fact_id``/``for``/``score``, score rounded to 4 decimals)
    plus ``method``: ``"rerank"`` (cross-encoder decided), ``"bge"``
    (dense top-1, re-ranker unavailable) or ``"keyword_tiebreak"``
    (re-rank top-2 within 0.01 → keyword top-1 wins).
    """

    EMBED_ID = "BAAI/bge-small-en-v1.5"
    RERANK_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    BATCH_SIZE = 128
    TOP_K = 10
    TIEBREAK_GAP = 0.01
    MAX_LENGTH = 512

    def __init__(self, fact_base: list[dict], *, device: str = "cpu",
                 seed: int = 0, pool_width: int | None = None,
                 tiebreak_gap: float | None = None):
        if not fact_base:
            raise ValueError("fact_base must be non-empty")
        self.base = fact_base
        self.device = device
        # rerank candidate-pool width; the class default (TOP_K=10) is the
        # registered basis for every measured number. The m365 diag
        # (2026-10-12) showed the width-10 pool hides the gold from the
        # reranker entirely on mq-010/023/040 (bge ranks 44/13/37, all
        # rerank #1 at width 50) — callers opt into the wider pool.
        self.pool_width = pool_width or self.TOP_K
        # keyword-tiebreak gap (top-2 rerank scores within this -> keyword
        # decides). The default is the registered basis, but on near-dupe
        # pairs (m365 holdout topics) the tie hands the decision to the
        # keyword leg, which overrode correct rank-1s 5x — tiebreak_gap=0
        # disables the tiebreak (pure rerank).
        self.tiebreak_gap = (self.TIEBREAK_GAP if tiebreak_gap is None
                             else tiebreak_gap)
        self.keyword = KeywordRetriever(fact_base)
        torch.manual_seed(seed)  # eval-mode forwards draw no samples; the
        # seed pins the (already deterministic) run against future edits.
        self._tok = _resolve(AutoTokenizer, self.EMBED_ID)
        self._embedder = _resolve(AutoModel, self.EMBED_ID).to(device).eval()
        try:
            self._rtok = _resolve(AutoTokenizer, self.RERANK_ID)
            self._reranker = _resolve(
                AutoModelForSequenceClassification, self.RERANK_ID
            ).to(device).eval()
        except Exception:  # degrade to bge-only, never raise (gate C3 style)
            self._rtok, self._reranker = None, None
        self._fact_embs: torch.Tensor | None = None

    # -- embedding ---------------------------------------------------------

    @torch.no_grad()
    def _embed(self, texts: list[str], *, prefix: str = "") -> torch.Tensor:
        """CLS-pooled, L2-normalized embeddings, fixed batch order."""
        out = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            enc = self._tok([prefix + t for t in texts[i:i + self.BATCH_SIZE]],
                            return_tensors="pt", padding=True,
                            truncation=True, max_length=self.MAX_LENGTH)
            cls = self._embedder(**enc.to(self.device)).last_hidden_state[:, 0]
            out.append(F.normalize(cls, dim=-1))
        return torch.cat(out)

    def _facts_matrix(self) -> torch.Tensor:
        if self._fact_embs is None:  # cached in-memory on first retrieve
            self._fact_embs = self._embed([f["fact"] for f in self.base])
        return self._fact_embs

    # -- re-rank -----------------------------------------------------------

    @torch.no_grad()
    def _rerank_scores(self, question: str, cand: list[int]) -> list[float]:
        enc = self._rtok([question] * len(cand),
                         [self.base[i]["fact"] for i in cand],
                         return_tensors="pt", padding=True, truncation=True,
                         max_length=self.MAX_LENGTH)
        logits = self._reranker(**enc.to(self.device)).logits
        return [float(x) for x in logits[:, 0]]

    # -- retrieval ---------------------------------------------------------

    def _hit(self, i: int, score: float, method: str) -> dict:
        f = self.base[i]
        return {"fact": f["fact"], "fact_id": f["id"], "for": f["for"],
                "score": round(score, 4), "method": method}

    def retrieve(self, question: str) -> dict:
        q = self._embed([question], prefix=self.QUERY_PREFIX)[0]
        sims = self._facts_matrix() @ q
        k = min(self.pool_width, len(self.base))
        cand = torch.topk(sims, k).indices.tolist()
        if self._reranker is None:
            i = cand[0]
            return self._hit(i, float(sims[i]), "bge")
        scores = self._rerank_scores(question, cand)
        order = sorted(range(k), key=lambda j: (-scores[j], j))  # ties: bge
        top, second = order[0], (order[1] if k > 1 else None)
        if (second is not None and self.tiebreak_gap > 0
                and scores[top] - scores[second] < self.tiebreak_gap):
            kw = self.keyword.retrieve(question)
            return {**kw, "method": "keyword_tiebreak"}
        return self._hit(cand[top], scores[top], "rerank")

    def retrieve_topk(self, question: str, k: int = 3) -> list[dict]:
        """Top-k facts in re-rank order (no keyword tiebreak — the k-list is
        the deliverable). Re-ranks the full TOP_K=10 candidate pool and
        returns its best k: measured hit@3 0.78 vs hit@1 0.59 on the
        near-duplicate-saturated PowerShell table motivates inject-top-k."""
        if k <= 0:
            raise ValueError("k must be >= 1")
        q = self._embed([question], prefix=self.QUERY_PREFIX)[0]
        sims = self._facts_matrix() @ q
        wide = min(max(self.pool_width, k), len(self.base))
        cand = torch.topk(sims, wide).indices.tolist()
        if self._reranker is None:
            order = list(range(wide))  # bge order
            top_scores = [float(sims[cand[j]]) for j in order]
        else:
            scores = self._rerank_scores(question, cand)
            order = sorted(range(wide), key=lambda j: (-scores[j], j))
            top_scores = [scores[j] for j in order]
        return [{**self._hit(cand[j], top_scores[r], "rerank_topk"),
                 "rank": r} for r, j in enumerate(order[:k])]
