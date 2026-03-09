"""
vectorstore.py
--------------
Persists and queries a :class:`~index.CodeIndex` using ChromaDB.

Each *chunk* becomes one ChromaDB document.  The chunk text is embedded by
Chroma's default embedding function (all-MiniLM-L6-v2 via sentence-
transformers) unless you pass your own ``embedding_function``.

Collections
-----------
One persistent Chroma client is created at ``persist_directory``.  Inside it
two collections are maintained:

``{collection_name}_chunks``
    One document per chunk.  Document = chunk text.  Metadata carries
    everything needed to reconstruct a :class:`~chunker.Chunk` and link back
    to its :class:`~index.FileEntry`.

``{collection_name}_meta``
    A single document that stores the index-level metadata (root path,
    created_at, INDEX_VERSION, per-file symbol lists) as a JSON string so the
    full :class:`~index.CodeIndex` can be round-tripped without a separate
    JSON file.

Public API
----------
``save_to_chroma(index, persist_dir, collection_name, embedding_function)``
    Upsert all chunks into Chroma.

``load_from_chroma(persist_dir, collection_name, embedding_function)``
    Reconstruct a :class:`~index.CodeIndex` from Chroma.

``query_chroma(persist_dir, prompt, collection_name, top_k, ...)``
    Semantic similarity search — returns :class:`~query.SearchResult` objects
    so it is a drop-in replacement for :func:`query.search`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# ChromaDB import (required)
# ---------------------------------------------------------------------------

try:
    import chromadb                                    # type: ignore
    from chromadb.config import Settings               # type: ignore
except ImportError as _exc:
    raise ImportError(
        "chromadb is not installed. Run:  pip install chromadb"
    ) from _exc

from index import CodeIndex, FileEntry, INDEX_VERSION
from chunker import Chunk
from parser import Symbol
from query import SearchResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CHUNKS_SUFFIX = "_chunks"
_META_SUFFIX   = "_meta"
_META_DOC_ID   = "__index_meta__"


def _chunk_metadata(chunk: Chunk, entry: FileEntry) -> dict[str, Any]:
    """Flatten chunk + file-entry fields into a Chroma-compatible metadata dict.

    Chroma metadata values must be str | int | float | bool.
    Lists (symbol_names) are JSON-encoded into a string.
    """
    return {
        # chunk fields
        "chunk_id":     chunk.chunk_id,
        "rel_path":     chunk.rel_path,
        "language":     entry.language,
        "start_line":   chunk.start_line,
        "end_line":     chunk.end_line,
        "token_count":  chunk.token_count,
        "chunk_index":  chunk.chunk_index,
        "total_chunks": chunk.total_chunks,
        "symbol_names": json.dumps(chunk.symbol_names),   # list → str
        # file fields
        "size_bytes":   entry.size_bytes,
        "parser_used":  entry.parser_used,
        "parse_error":  entry.parse_error,
    }


def _metadata_to_chunk(meta: dict[str, Any], text: str) -> Chunk:
    return Chunk(
        chunk_id     = meta["chunk_id"],
        rel_path     = meta["rel_path"],
        language     = meta["language"],
        start_line   = int(meta["start_line"]),
        end_line     = int(meta["end_line"]),
        token_count  = int(meta["token_count"]),
        chunk_index  = int(meta["chunk_index"]),
        total_chunks = int(meta["total_chunks"]),
        symbol_names = json.loads(meta.get("symbol_names", "[]")),
        text         = text,
    )


def _metadata_to_entry_stub(meta: dict[str, Any]) -> FileEntry:
    """Create a FileEntry with no symbols — symbols come from the meta collection."""
    return FileEntry(
        rel_path    = meta["rel_path"],
        language    = meta["language"],
        size_bytes  = int(meta["size_bytes"]),
        parser_used = meta.get("parser_used", ""),
        parse_error = meta.get("parse_error", ""),
        symbols     = [],
        chunks      = [],
    )


def _get_client(persist_dir: str | Path) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=str(Path(persist_dir).resolve()),
        settings=Settings(anonymized_telemetry=False),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_to_chroma(
    index: CodeIndex,
    persist_dir: str | Path,
    collection_name: str = "code_index",
    embedding_function: Any | None = None,
    batch_size: int = 100,
    verbose: bool = False,
) -> None:
    """
    Upsert all chunks from *index* into a ChromaDB persistent collection.

    Parameters
    ----------
    index:
        The :class:`~index.CodeIndex` to store.
    persist_dir:
        Directory where Chroma stores its SQLite + vector data.
        Created automatically if it does not exist.
    collection_name:
        Base name for the Chroma collections.
        Two collections are created: ``<name>_chunks`` and ``<name>_meta``.
    embedding_function:
        Any Chroma-compatible embedding function.  ``None`` → Chroma's
        default (``all-MiniLM-L6-v2``).
    batch_size:
        Number of documents upserted per Chroma API call.
    verbose:
        Print progress.
    """
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = _get_client(persist_dir)

    ef_kwargs: dict[str, Any] = {}
    if embedding_function is not None:
        ef_kwargs["embedding_function"] = embedding_function

    chunks_col = client.get_or_create_collection(
        name=collection_name + _CHUNKS_SUFFIX,
        metadata={"hnsw:space": "cosine"},
        **ef_kwargs,
    )
    meta_col = client.get_or_create_collection(
        name=collection_name + _META_SUFFIX,
        **ef_kwargs,
    )

    # ---- upsert chunks ----
    all_chunks: list[tuple[Chunk, FileEntry]] = [
        (chunk, entry)
        for entry in index.files
        for chunk in entry.chunks
    ]

    total = len(all_chunks)
    if verbose:
        print(f"  Upserting {total} chunks into '{collection_name + _CHUNKS_SUFFIX}' …")

    for batch_start in range(0, total, batch_size):
        batch = all_chunks[batch_start : batch_start + batch_size]
        chunks_col.upsert(
            ids        = [c.chunk_id for c, _ in batch],
            documents  = [c.text     for c, _ in batch],
            metadatas  = [_chunk_metadata(c, e) for c, e in batch],
        )
        if verbose:
            done = min(batch_start + batch_size, total)
            print(f"    {done}/{total}")

    # ---- upsert index-level metadata ----
    # Store symbol lists + index header as a JSON blob in the meta collection.
    meta_payload = {
        "version":    INDEX_VERSION,
        "root":       index.root,
        "created_at": index.created_at,
        "files": [
            {
                "rel_path":    e.rel_path,
                "language":    e.language,
                "size_bytes":  e.size_bytes,
                "parser_used": e.parser_used,
                "parse_error": e.parse_error,
                "symbols": [
                    {
                        "name":       s.name,
                        "kind":       s.kind,
                        "start_line": s.start_line,
                        "end_line":   s.end_line,
                        "signature":  s.signature,
                        "docstring":  s.docstring,
                        "parent":     s.parent,
                    }
                    for s in e.symbols
                ],
            }
            for e in index.files
        ],
    }

    meta_col.upsert(
        ids       = [_META_DOC_ID],
        documents = [json.dumps(meta_payload, ensure_ascii=False)],
        metadatas = [{"type": "index_meta"}],
    )

    if verbose:
        stats = index.stats
        print(
            f"\n  ✓ Saved to ChromaDB at '{persist_dir}'\n"
            f"    collection : {collection_name}\n"
            f"    files      : {stats['files']}\n"
            f"    symbols    : {stats['symbols']}\n"
            f"    chunks     : {stats['chunks']}\n"
        )


def load_from_chroma(
    persist_dir: str | Path,
    collection_name: str = "code_index",
    embedding_function: Any | None = None,
) -> CodeIndex:
    """
    Reconstruct a :class:`~index.CodeIndex` from ChromaDB.

    Parameters
    ----------
    persist_dir:
        Directory passed to :func:`save_to_chroma`.
    collection_name:
        Same base name used when saving.
    embedding_function:
        Must match the function used when saving (if non-default).
    """
    client = _get_client(persist_dir)

    ef_kwargs: dict[str, Any] = {}
    if embedding_function is not None:
        ef_kwargs["embedding_function"] = embedding_function

    meta_col = client.get_collection(
        name=collection_name + _META_SUFFIX,
        **ef_kwargs,
    )
    chunks_col = client.get_collection(
        name=collection_name + _CHUNKS_SUFFIX,
        **ef_kwargs,
    )

    # ---- load metadata ----
    meta_result = meta_col.get(ids=[_META_DOC_ID], include=["documents"])
    if not meta_result["documents"]:
        raise ValueError(f"No index metadata found in collection '{collection_name + _META_SUFFIX}'")

    meta_payload = json.loads(meta_result["documents"][0])
    index = CodeIndex(
        root       = meta_payload["root"],
        created_at = meta_payload["created_at"],
    )

    # Build a lookup: rel_path → list[Symbol]
    symbols_by_path: dict[str, list[Symbol]] = {}
    entry_meta: dict[str, dict] = {}
    for f in meta_payload.get("files", []):
        path = f["rel_path"]
        symbols_by_path[path] = [
            Symbol(
                name           = s["name"],
                kind           = s["kind"],
                start_line     = s["start_line"],
                end_line       = s["end_line"],
                signature      = s["signature"],
                docstring      = s.get("docstring", ""),
                parent         = s.get("parent", ""),
                source_snippet = "",
            )
            for s in f.get("symbols", [])
        ]
        entry_meta[path] = f

    # ---- load all chunks ----
    all_chunks_result = chunks_col.get(include=["documents", "metadatas"])
    chunks_by_path: dict[str, list[Chunk]] = {}
    for doc, meta in zip(
        all_chunks_result["documents"],
        all_chunks_result["metadatas"],
    ):
        chunk = _metadata_to_chunk(meta, doc)
        chunks_by_path.setdefault(chunk.rel_path, []).append(chunk)

    # Sort chunks by chunk_index within each file
    for path in chunks_by_path:
        chunks_by_path[path].sort(key=lambda c: c.chunk_index)

    # ---- reassemble FileEntry objects ----
    for path, em in entry_meta.items():
        index.files.append(FileEntry(
            rel_path    = path,
            language    = em["language"],
            size_bytes  = em["size_bytes"],
            parser_used = em.get("parser_used", ""),
            parse_error = em.get("parse_error", ""),
            symbols     = symbols_by_path.get(path, []),
            chunks      = chunks_by_path.get(path, []),
        ))

    return index


def query_chroma(
    persist_dir: str | Path,
    prompt: str,
    collection_name: str = "code_index",
    top_k: int = 10,
    language_filter: str | None = None,
    embedding_function: Any | None = None,
) -> list[SearchResult]:
    """
    Semantic similarity search over the ChromaDB chunk collection.

    Returns a list of :class:`~query.SearchResult` objects (same shape as
    :func:`query.search`) so callers need not care which backend is used.

    Parameters
    ----------
    persist_dir:
        ChromaDB persistence directory.
    prompt:
        Natural-language query.
    collection_name:
        Base collection name used when saving.
    top_k:
        Maximum number of results.
    language_filter:
        When set, restrict results to the given language (e.g. ``"python"``).
    embedding_function:
        Must match the function used when saving.
    """
    client = _get_client(persist_dir)

    ef_kwargs: dict[str, Any] = {}
    if embedding_function is not None:
        ef_kwargs["embedding_function"] = embedding_function

    chunks_col = client.get_collection(
        name=collection_name + _CHUNKS_SUFFIX,
        **ef_kwargs,
    )

    where: dict[str, Any] | None = None
    if language_filter:
        where = {"language": {"$eq": language_filter}}

    query_kwargs: dict[str, Any] = dict(
        query_texts = [prompt],
        n_results   = top_k,
        include     = ["documents", "metadatas", "distances"],
    )
    if where:
        query_kwargs["where"] = where

    results = chunks_col.query(**query_kwargs)

    # Chroma returns lists-of-lists (one per query_text)
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]   # cosine distance: lower = more similar

    search_results: list[SearchResult] = []
    for doc, meta, dist in zip(docs, metadatas, distances):
        chunk = _metadata_to_chunk(meta, doc)
        entry = _metadata_to_entry_stub(meta)

        # Convert cosine distance [0, 2] → similarity score (higher = better)
        score = max(0.0, 1.0 - dist)

        # surface keywords that overlap between prompt and chunk text
        prompt_words = set(re.findall(r"\b\w{3,}\b", prompt.lower()))
        chunk_words  = set(re.findall(r"\b\w{3,}\b", doc.lower()))
        matched      = sorted(prompt_words & chunk_words)

        search_results.append(SearchResult(
            chunk            = chunk,
            file_entry       = entry,
            score            = score,
            matched_keywords = matched,
        ))

    # already sorted by distance, but re-sort by score descending to be safe
    search_results.sort(key=lambda r: r.score, reverse=True)
    return search_results
