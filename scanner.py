"""
scanner.py
----------
Walks a root directory and collects Python source files.
Supports include/exclude glob patterns and prompt-based relevance filtering.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SourceFile:
    """Represents a single discovered source file."""
    path: Path
    rel_path: str          # relative to scan root
    size_bytes: int
    content: str = field(repr=False, default="")

    @property
    def language(self) -> str:
        suffix = self.path.suffix.lower()
        return _SUFFIX_TO_LANG.get(suffix, "unknown")


_SUFFIX_TO_LANG: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".jsx":  "javascript",
    ".tsx":  "typescript",
    ".c":    "c",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".h":    "c",
    ".hpp":  "cpp",
    ".rs":   "rust",
    ".go":   "go",
    ".java": "java",
    ".rb":   "ruby",
}

_DEFAULT_INCLUDE_EXTS: frozenset[str] = frozenset(_SUFFIX_TO_LANG.keys())

_DEFAULT_EXCLUDES: list[str] = [
    "**/__pycache__/**",
    "**/.git/**",
    "**/node_modules/**",
    "**/.venv/**",
    "**/venv/**",
    "**/dist/**",
    "**/build/**",
    "**/*.egg-info/**",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_directory(
    root: str | Path,
    *,
    include_exts: frozenset[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_file_size_kb: int = 512,
    prompt_keywords: list[str] | None = None,
) -> list[SourceFile]:
    """
    Recursively scan *root* and return a list of :class:`SourceFile` objects.

    Parameters
    ----------
    root:
        Directory to scan.
    include_exts:
        Set of file extensions to include (with leading dot, e.g. ``{'.py'}``).
        Defaults to all supported languages.
    exclude_patterns:
        Glob patterns (relative to root) to skip.  Defaults to common junk
        directories such as ``__pycache__``, ``.git``, ``node_modules``, etc.
    max_file_size_kb:
        Files larger than this are skipped (they are likely generated/binary).
    prompt_keywords:
        When provided the scanner applies a fast *keyword pre-filter*: only
        files whose path or content contain at least one keyword are returned.
        This keeps the symbol-extraction stage focused on relevant files.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    include_exts = include_exts or _DEFAULT_INCLUDE_EXTS
    exclude_patterns = exclude_patterns or _DEFAULT_EXCLUDES
    max_bytes = max_file_size_kb * 1024

    results: list[SourceFile] = []

    for src_file in _walk(root, include_exts, exclude_patterns, max_bytes):
        if prompt_keywords and not _keyword_match(src_file, prompt_keywords):
            continue
        results.append(src_file)

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _walk(
    root: Path,
    include_exts: frozenset[str],
    exclude_patterns: list[str],
    max_bytes: int,
) -> Iterator[SourceFile]:
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirpath_p = Path(dirpath)
        rel_dir = dirpath_p.relative_to(root)

        # Prune excluded directories in-place so os.walk skips them entirely
        dirnames[:] = [
            d for d in dirnames
            if not _is_excluded(rel_dir / d, exclude_patterns)
        ]

        for fname in sorted(filenames):
            fpath = dirpath_p / fname
            rel_path = fpath.relative_to(root)

            if fpath.suffix.lower() not in include_exts:
                continue
            if _is_excluded(rel_path, exclude_patterns):
                continue

            try:
                size = fpath.stat().st_size
            except OSError:
                continue

            if size > max_bytes:
                continue

            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            yield SourceFile(
                path=fpath,
                rel_path=str(rel_path),
                size_bytes=size,
                content=content,
            )


def _is_excluded(rel_path: Path, patterns: list[str]) -> bool:
    rel_str = str(rel_path).replace("\\", "/")
    for pat in patterns:
        # strip leading **/ for simple suffix matching
        if fnmatch.fnmatch(rel_str, pat):
            return True
        # also match just the final component
        if fnmatch.fnmatch(rel_path.name, pat.lstrip("*/")):
            return True
    return False


def _keyword_match(src_file: SourceFile, keywords: list[str]) -> bool:
    """Return True if *any* keyword appears in the file path or content (case-insensitive)."""
    haystack_path = src_file.rel_path.lower()
    haystack_content = src_file.content.lower()
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in haystack_path or kw_lower in haystack_content:
            return True
    return False
