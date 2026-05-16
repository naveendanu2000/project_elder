"""
chunker.py
----------
Splits large source files / symbol bodies into token-budget chunks of
≤ 400 tokens each, while keeping logical boundaries intact.

"Token" estimation: 1 token ≈ 4 characters (a common heuristic that works
well for code).  If tiktoken is installed we use it for more accurate counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from parser import ParsedFile, Symbol


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4          # fallback heuristic
_TIKTOKEN_ENC = None
_TIKTOKEN_TRIED = False


def _count_tokens(text: str) -> int:
    global _TIKTOKEN_ENC, _TIKTOKEN_TRIED
    if not _TIKTOKEN_TRIED:
        _TIKTOKEN_TRIED = True
        try:
            import tiktoken  # type: ignore
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass

    if _TIKTOKEN_ENC is not None:
        return len(_TIKTOKEN_ENC.encode(text))
    return max(1, len(text) // _CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A token-budget slice of a source file, ready for embedding / search."""
    chunk_id: str         # "<rel_path>#chunk<N>"
    rel_path: str
    language: str
    start_line: int
    end_line: int
    token_count: int
    text: str = field(repr=False)
    symbol_names: list[str] = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MAX_TOKENS_PER_CHUNK = 400


def chunk_parsed_file(
    parsed: ParsedFile,
    source_content: str,
    max_tokens: int = MAX_TOKENS_PER_CHUNK,
) -> list[Chunk]:
    """
    Produce a list of :class:`Chunk` objects for *parsed*.

    Strategy
    --------
    1. If the file fits in one chunk → return it as-is.
    2. Otherwise split on *symbol boundaries* first (keeps functions/classes
       intact wherever possible).
    3. Any individual symbol that still exceeds *max_tokens* is split on blank
       lines, then on individual lines as a last resort.
    """
    total_tokens = _count_tokens(source_content)
    if total_tokens <= max_tokens:
        end_line = max(1, len(source_content.splitlines()))
        symbol_names = [sym.name for sym in parsed.symbols]
        return [_make_chunk(
            parsed.rel_path,
            parsed.language,
            source_content,
            1,
            end_line,
            0,
            symbol_names,
        )]

    # --- split by symbol boundaries ---
    raw_chunks = list(_split_by_symbols(parsed, source_content, max_tokens))

    # number them
    chunks: list[Chunk] = []
    total = len(raw_chunks)
    for idx, (start, end, text, sym_names) in enumerate(raw_chunks):
        c = Chunk(
            chunk_id=f"{parsed.rel_path}#chunk{idx + 1}",
            rel_path=parsed.rel_path,
            language=parsed.language,
            start_line=start,
            end_line=end,
            token_count=_count_tokens(text),
            text=text,
            symbol_names=sym_names,
            chunk_index=idx,
            total_chunks=total,
        )
        chunks.append(c)
    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    rel_path: str,
    language: str,
    text: str,
    start_line: int,
    end_line: int,
    index: int,
    symbol_names: list[str] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"{rel_path}#chunk{index + 1}",
        rel_path=rel_path,
        language=language,
        start_line=start_line,
        end_line=end_line,
        token_count=_count_tokens(text),
        text=text,
        symbol_names=symbol_names or [],
        chunk_index=index,
        total_chunks=1,  # will be patched by caller
    )


def _split_by_symbols(
    parsed: ParsedFile,
    source_content: str,
    max_tokens: int,
) -> Iterator[tuple[int, int, str, list[str]]]:
    """
    Yield ``(start_line, end_line, text, symbol_names)`` tuples.

    We greedily pack consecutive symbols into a chunk until we exceed
    *max_tokens*, then flush and start a new chunk.
    """
    lines = source_content.splitlines(keepends=True)
    n_lines = len(lines)

    # Build a map: line_number (1-based) → Symbol
    sym_by_start: dict[int, Symbol] = {}
    for sym in sorted(parsed.symbols, key=lambda s: s.start_line):
        sym_by_start[sym.start_line] = sym

    # Cursor-based greedy packing
    cursor = 1        # 1-based current line
    buf_lines: list[str] = []
    buf_start = 1
    buf_syms: list[str] = []
    buf_tokens = 0

    def _flush(end_line: int) -> tuple[int, int, str, list[str]]:
        return buf_start, end_line, "".join(buf_lines), list(buf_syms)

    while cursor <= n_lines:
        sym = sym_by_start.get(cursor)

        if sym:
            sym_lines = lines[sym.start_line - 1: sym.end_line]
            sym_text = "".join(sym_lines)
            sym_tokens = _count_tokens(sym_text)

            # flush current buffer if adding this symbol would overflow
            if buf_tokens + sym_tokens > max_tokens and buf_lines:
                yield _flush(cursor - 1)
                buf_lines = []
                buf_start = cursor
                buf_syms = []
                buf_tokens = 0

            # symbol itself too big → sub-split it
            if sym_tokens > max_tokens:
                if buf_lines:
                    yield _flush(sym.start_line - 1)
                    buf_lines = []
                    buf_start = sym.start_line
                    buf_syms = []
                    buf_tokens = 0

                for sub_start, sub_end, sub_text, _ in _split_text(
                    sym_text, sym.start_line, max_tokens
                ):
                    yield sub_start, sub_end, sub_text, [sym.name]

                cursor = sym.end_line + 1
                buf_start = cursor
                continue

            buf_lines += sym_lines
            buf_syms.append(sym.name)
            buf_tokens += sym_tokens
            cursor = sym.end_line + 1

        else:
            # line not part of any known symbol
            line_text = lines[cursor - 1]
            line_tokens = _count_tokens(line_text)
            if buf_tokens + line_tokens > max_tokens and buf_lines:
                yield _flush(cursor - 1)
                buf_lines = []
                buf_start = cursor
                buf_syms = []
                buf_tokens = 0

            buf_lines.append(line_text)
            buf_tokens += line_tokens
            cursor += 1

    if buf_lines:
        yield _flush(cursor - 1)


def _split_text(
    text: str,
    base_line: int,
    max_tokens: int,
) -> Iterator[tuple[int, int, str, list[str]]]:
    """
    Last-resort splitter: chunk raw text by blank-line boundaries, then by
    individual lines if still too large.
    """
    paragraphs = re.split(r"(\n\s*\n)", text)
    buf: list[str] = []
    buf_start = base_line
    buf_tokens = 0
    cur_line = base_line

    for para in paragraphs:
        p_tokens = _count_tokens(para)
        if buf_tokens + p_tokens > max_tokens and buf:
            joined = "".join(buf)
            yield buf_start, cur_line - 1, joined, []
            buf = []
            buf_start = cur_line
            buf_tokens = 0

        if p_tokens > max_tokens:
            # split line by line
            for line in para.splitlines(keepends=True):
                lt = _count_tokens(line)
                if buf_tokens + lt > max_tokens and buf:
                    joined = "".join(buf)
                    yield buf_start, cur_line - 1, joined, []
                    buf = []
                    buf_start = cur_line
                    buf_tokens = 0
                buf.append(line)
                buf_tokens += lt
                cur_line += 1
        else:
            buf.append(para)
            buf_tokens += p_tokens
            cur_line += para.count("\n")

    if buf:
        yield buf_start, cur_line, "".join(buf), []
