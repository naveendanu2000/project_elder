"""
index.py
--------
Assembles the final index from scanned files, parsed symbols, and chunks,
then persists / loads the index as JSON.

Index schema (saved to disk)
-----------------------------
{
  "version": "1.0",
  "root": "/absolute/path/to/project",
  "created_at": "ISO-8601 timestamp",
  "stats": { "files": N, "symbols": N, "chunks": N },
  "files": [
    {
      "rel_path": "src/foo.py",
      "language": "python",
      "size_bytes": 1234,
      "parser_used": "treesitter",
      "parse_error": "",
      "symbols": [
        {
          "name": "MyClass",
          "kind": "class",
          "start_line": 10,
          "end_line": 40,
          "signature": "class MyClass(Base):",
          "docstring": "...",
          "parent": ""
        },
        ...
      ],
      "chunks": [
        {
          "chunk_id": "src/foo.py#chunk1",
          "start_line": 1,
          "end_line": 55,
          "token_count": 387,
          "symbol_names": ["MyClass", "my_func"],
          "chunk_index": 0,
          "total_chunks": 2,
          "text": "..."
        },
        ...
      ]
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scanner import SourceFile
from parser import ParsedFile, Symbol, parse_file
from chunker import Chunk, chunk_parsed_file


INDEX_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileEntry:
    rel_path: str
    language: str
    size_bytes: int
    parser_used: str
    parse_error: str
    symbols: list[Symbol]
    chunks: list[Chunk]


@dataclass
class CodeIndex:
    root: str
    created_at: str
    files: list[FileEntry] = field(default_factory=list)

    # convenience
    @property
    def stats(self) -> dict[str, int]:
        return {
            "files":   len(self.files),
            "symbols": sum(len(f.symbols) for f in self.files),
            "chunks":  sum(len(f.chunks) for f in self.files),
        }


# ---------------------------------------------------------------------------
# Building the index
# ---------------------------------------------------------------------------

def build_index(
    root: str | Path,
    source_files: list[SourceFile],
    max_tokens_per_chunk: int = 400,
    verbose: bool = False,
) -> CodeIndex:
    """
    Parse all *source_files* and assemble a :class:`CodeIndex`.

    Parameters
    ----------
    root:
        Original scan root (stored as metadata).
    source_files:
        Output of :func:`scanner.scan_directory`.
    max_tokens_per_chunk:
        Forwarded to :func:`chunker.chunk_parsed_file`.
    verbose:
        Print progress to stdout.
    """
    index = CodeIndex(
        root=str(Path(root).resolve()),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    total = len(source_files)
    for i, src in enumerate(source_files, start=1):
        if verbose:
            print(f"  [{i}/{total}] parsing  {src.rel_path}")

        parsed: ParsedFile = parse_file(src)
        chunks: list[Chunk] = chunk_parsed_file(parsed, src.content, max_tokens_per_chunk)

        # patch total_chunks now that we know the real count
        for c in chunks:
            c.total_chunks = len(chunks)

        entry = FileEntry(
            rel_path=src.rel_path,
            language=src.language,
            size_bytes=src.size_bytes,
            parser_used=parsed.parser_used,
            parse_error=parsed.parse_error,
            symbols=parsed.symbols,
            chunks=chunks,
        )
        index.files.append(entry)

    return index


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _symbol_to_dict(sym: Symbol) -> dict[str, Any]:
    return {
        "name":           sym.name,
        "kind":           sym.kind,
        "start_line":     sym.start_line,
        "end_line":       sym.end_line,
        "signature":      sym.signature,
        "docstring":      sym.docstring,
        "parent":         sym.parent,
    }


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id":     chunk.chunk_id,
        "start_line":   chunk.start_line,
        "end_line":     chunk.end_line,
        "token_count":  chunk.token_count,
        "symbol_names": chunk.symbol_names,
        "chunk_index":  chunk.chunk_index,
        "total_chunks": chunk.total_chunks,
        "text":         chunk.text,
    }


def _entry_to_dict(entry: FileEntry) -> dict[str, Any]:
    return {
        "rel_path":    entry.rel_path,
        "language":    entry.language,
        "size_bytes":  entry.size_bytes,
        "parser_used": entry.parser_used,
        "parse_error": entry.parse_error,
        "symbols":     [_symbol_to_dict(s) for s in entry.symbols],
        "chunks":      [_chunk_to_dict(c) for c in entry.chunks],
    }


def save_index(index: CodeIndex, output_path: str | Path, indent: int = 2) -> Path:
    """Serialise *index* to a JSON file at *output_path*."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version":    INDEX_VERSION,
        "root":       index.root,
        "created_at": index.created_at,
        "stats":      index.stats,
        "files":      [_entry_to_dict(e) for e in index.files],
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, ensure_ascii=False)

    return output_path


def load_index(index_path: str | Path) -> CodeIndex:
    """
    Deserialise a JSON index file previously saved by :func:`save_index`.

    Returns a :class:`CodeIndex` with fully typed entries.
    """
    index_path = Path(index_path)
    with index_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    index = CodeIndex(root=raw["root"], created_at=raw["created_at"])

    for f in raw.get("files", []):
        symbols = [
            Symbol(
                name=s["name"],
                kind=s["kind"],
                start_line=s["start_line"],
                end_line=s["end_line"],
                signature=s["signature"],
                docstring=s.get("docstring", ""),
                parent=s.get("parent", ""),
                source_snippet="",   # not stored to save space
            )
            for s in f.get("symbols", [])
        ]
        chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                rel_path=f["rel_path"],
                language=f["language"],
                start_line=c["start_line"],
                end_line=c["end_line"],
                token_count=c["token_count"],
                text=c["text"],
                symbol_names=c.get("symbol_names", []),
                chunk_index=c["chunk_index"],
                total_chunks=c["total_chunks"],
            )
            for c in f.get("chunks", [])
        ]
        index.files.append(FileEntry(
            rel_path=f["rel_path"],
            language=f["language"],
            size_bytes=f["size_bytes"],
            parser_used=f.get("parser_used", ""),
            parse_error=f.get("parse_error", ""),
            symbols=symbols,
            chunks=chunks,
        ))

    return index
