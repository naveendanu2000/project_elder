"""
query.py
--------
Given a natural-language prompt and a loaded :class:`~index.CodeIndex`,
returns the most relevant :class:`~chunker.Chunk` objects.

Ranking strategy (no external vector DB needed)
------------------------------------------------
1. **Keyword overlap** – BM25-style TF*IDF approximation over chunk text +
   symbol names.
2. **Symbol-kind boost** – chunks containing a symbol whose name closely
   matches a prompt word get a bonus.
3. **File-path boost** – files whose path contains a prompt keyword rank higher.

For production use you can swap the scorer for a real embedding search
(e.g. sentence-transformers + FAISS) without changing the public API.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

from index import CodeIndex, FileEntry
from chunker import Chunk


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    chunk: Chunk
    file_entry: FileEntry
    score: float
    matched_keywords: list[str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(
    index: CodeIndex,
    prompt: str,
    *,
    top_k: int = 10,
    min_score: float = 0.01,
    language_filter: str | None = None,
) -> list[SearchResult]:
    """
    Return up to *top_k* chunks most relevant to *prompt*.

    Parameters
    ----------
    index:
        A loaded :class:`~index.CodeIndex`.
    prompt:
        Natural-language query, e.g. ``"how does authentication work"``.
    top_k:
        Maximum number of results.
    min_score:
        Chunks scoring below this threshold are excluded.
    language_filter:
        Optional language name (e.g. ``"python"``) to restrict results.
    """
    keywords = _extract_keywords(prompt)
    if not keywords:
        return []

    # Build corpus for IDF
    all_chunks: list[tuple[Chunk, FileEntry]] = []
    for entry in index.files:
        if language_filter and entry.language != language_filter:
            continue
        for chunk in entry.chunks:
            all_chunks.append((chunk, entry))

    if not all_chunks:
        return []

    idf = _compute_idf(keywords, all_chunks)
    results: list[SearchResult] = []

    for chunk, entry in all_chunks:
        score, matched = _score_chunk(chunk, entry, keywords, idf)
        if score >= min_score:
            results.append(SearchResult(
                chunk=chunk,
                file_entry=entry,
                score=score,
                matched_keywords=matched,
            ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def format_results(results: list[SearchResult], *, show_text: bool = True) -> str:
    """Human-readable summary of search results."""
    if not results:
        return "No relevant code found."

    lines: list[str] = []
    for rank, r in enumerate(results, start=1):
        c = r.chunk
        header = (
            f"#{rank}  score={r.score:.3f}  "
            f"{c.rel_path}  lines {c.start_line}-{c.end_line}  "
            f"[{', '.join(c.symbol_names) or 'no symbols'}]  "
            f"keywords: {', '.join(r.matched_keywords)}"
        )
        lines.append(header)
        if show_text:
            lines.append("-" * 60)
            lines.append(c.text[:1200])  # cap display length
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset("""
a about an are as at be been by do does for from has have how i
if in into is it its just my no not of on or should so that the
their them then there these they this to was were what when where
which who will with would you your
""".split())

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _extract_keywords(prompt: str) -> list[str]:
    """Lower-case, de-camelcase, and deduplicate meaningful words from *prompt*."""
    # split on non-alphanumeric
    tokens = re.split(r"[^a-zA-Z0-9_]+", prompt)
    words: list[str] = []
    for tok in tokens:
        # split camelCase / PascalCase
        parts = _CAMEL_RE.sub(" ", tok).split()
        words.extend(p.lower() for p in parts)

    # also keep the original token as a phrase (helps match symbol names)
    for tok in tokens:
        if "_" in tok:
            words.append(tok.lower())

    keywords = [w for w in words if len(w) > 2 and w not in _STOP_WORDS]
    # deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _chunk_document(chunk: Chunk, entry: FileEntry) -> str:
    """Concatenate all searchable text for a chunk."""
    sym_text = " ".join(chunk.symbol_names)
    return f"{entry.rel_path} {sym_text} {chunk.text}".lower()


def _compute_idf(keywords: list[str], corpus: list[tuple[Chunk, FileEntry]]) -> dict[str, float]:
    N = len(corpus)
    df: dict[str, int] = defaultdict(int)
    for chunk, entry in corpus:
        doc = _chunk_document(chunk, entry)
        for kw in keywords:
            if kw in doc:
                df[kw] += 1
    return {
        kw: math.log((N + 1) / (df[kw] + 1)) + 1
        for kw in keywords
    }


def _score_chunk(
    chunk: Chunk,
    entry: FileEntry,
    keywords: list[str],
    idf: dict[str, float],
) -> tuple[float, list[str]]:
    doc = _chunk_document(chunk, entry)
    doc_tokens = re.split(r"\W+", doc)
    tf_counter = Counter(doc_tokens)
    doc_len = max(len(doc_tokens), 1)

    score = 0.0
    matched: list[str] = []

    for kw in keywords:
        tf = tf_counter.get(kw, 0)
        if tf == 0:
            continue
        matched.append(kw)
        # BM25-lite: TF saturation
        k1 = 1.5
        b  = 0.75
        avg_len = 200  # rough average chunk length in tokens
        norm_tf = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_len))
        score += norm_tf * idf.get(kw, 1.0)

    # bonus: symbol name exact match
    sym_names_lower = [s.lower() for s in chunk.symbol_names]
    for kw in keywords:
        if kw in sym_names_lower:
            score *= 1.5
            break

    # bonus: file path match
    path_lower = entry.rel_path.lower()
    for kw in keywords:
        if kw in path_lower:
            score *= 1.3
            break

    return score, matched
