"""
elder.py
--------
Friendly prompt-first CLI for asking questions about the current codebase.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from index import CodeIndex, build_index, load_index, save_index
from query import format_results, search
from scanner import scan_directory


CACHE_DIRNAME = ".elder"
INDEX_FILENAME = "index.json"
CHROMA_DIRNAME = "chroma"
COLLECTION_NAME = "code_index"


def _configure_console() -> None:
    """Prefer UTF-8 output on Windows consoles that default to cp1252."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elder",
        description="Ask natural-language questions about a codebase.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help='Prompt to answer, e.g. elder check the code',
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root to inspect (default: current directory).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "search", "openai", "claude", "ollama"],
        default="auto",
        help="Answer backend. 'search' uses local keyword search only.",
    )
    parser.add_argument("--api-key", default=None, help="API key for OpenAI or Claude.")
    parser.add_argument("--model", default=None, help="Model override for the selected backend.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of chunks to retrieve.")
    parser.add_argument("--language", default=None, help="Optional language filter, e.g. python.")
    parser.add_argument("--max-tokens", type=int, default=400, help="Chunk token budget.")
    parser.add_argument("--max-file-size-kb", type=int, default=512)
    parser.add_argument("--reindex", action="store_true", help="Rebuild the project cache.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print indexing progress.")
    return parser


def _cache_paths(root: Path) -> tuple[Path, Path]:
    cache_dir = root / CACHE_DIRNAME
    return cache_dir / INDEX_FILENAME, cache_dir / CHROMA_DIRNAME


def _needs_reindex(index_path: Path, root: Path) -> bool:
    if not index_path.is_file():
        return True

    index_mtime = index_path.stat().st_mtime
    try:
        sources = scan_directory(root)
    except OSError:
        return True

    return any(src.path.stat().st_mtime > index_mtime for src in sources)


def _ensure_index(args: argparse.Namespace, root: Path) -> tuple[CodeIndex, Path]:
    index_path, _ = _cache_paths(root)
    should_build = args.reindex or _needs_reindex(index_path, root)

    if not should_build:
        return load_index(index_path), index_path

    print(f"Indexing {root} ...")
    sources = scan_directory(
        root,
        max_file_size_kb=args.max_file_size_kb,
    )
    if not sources:
        raise SystemExit("No supported source files found.")

    index = build_index(
        root,
        sources,
        max_tokens_per_chunk=args.max_tokens,
        verbose=args.verbose,
    )
    save_index(index, index_path)
    stats = index.stats
    print(
        f"Indexed {stats['files']} files, {stats['symbols']} symbols, "
        f"{stats['chunks']} chunks."
    )
    return index, index_path


def _select_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    return "search"


def _print_sources(sources: list) -> None:
    if not sources:
        return

    print("\nSources")
    for src in sources:
        syms = ", ".join(src.symbols) or "no symbols"
        print(f"- {src.file}:{src.lines}  score={src.similarity}  [{syms}]")


def _looks_like_review(prompt: str) -> bool:
    words = {w.strip(".,!?;:").lower() for w in prompt.split()}
    return bool(words & {"check", "review", "audit", "inspect", "scan"})


def _print_project_overview(index: CodeIndex, limit: int) -> None:
    print("Review targets")
    for entry in index.files[:limit]:
        symbols = [
            symbol.name for symbol in entry.symbols
            if symbol.kind in {"class", "function", "method", "type", "interface"}
        ]
        summary = ", ".join(symbols[:8]) if symbols else "no named symbols"
        print(f"- {entry.rel_path} ({entry.language}): {summary}")


def _local_answer(index: CodeIndex, prompt: str, args: argparse.Namespace) -> None:
    stats = index.stats
    print("Local code search mode")
    print(
        f"Project has {stats['files']} indexed files, {stats['symbols']} symbols, "
        f"and {stats['chunks']} chunks."
    )

    parse_errors = [
        f for f in index.files
        if f.parse_error and f.parse_error != "Unsupported language"
    ]
    if parse_errors:
        print("\nParse issues")
        for entry in parse_errors[:10]:
            print(f"- {entry.rel_path}: {entry.parse_error}")

    results = search(
        index,
        prompt,
        top_k=args.top_k,
        language_filter=args.language,
    )

    is_review = _looks_like_review(prompt)
    if is_review and not parse_errors:
        print("\nNo parser-level problems were found in the indexed files.")

    if results:
        print("\nMost relevant code")
        print(format_results(results, show_text=True))
    elif is_review:
        print()
        _print_project_overview(index, args.top_k)
    else:
        print("\nMost relevant code")
        print("No relevant code found.")

    print(
        "\nFor full natural-language fixes/analysis, set OPENAI_API_KEY or "
        "ANTHROPIC_API_KEY, or run with --backend ollama."
    )


def _llm_answer(
    index: CodeIndex,
    prompt: str,
    args: argparse.Namespace,
    root: Path,
    backend: str,
) -> None:
    from rag import CodeRAG
    from vectorstore import save_to_chroma

    _, chroma_dir = _cache_paths(root)
    if args.reindex or not chroma_dir.exists():
        print(f"Preparing semantic index at {chroma_dir} ...")
        save_to_chroma(
            index,
            persist_dir=chroma_dir,
            collection_name=COLLECTION_NAME,
            verbose=args.verbose,
        )

    rag = CodeRAG(
        db_path=str(chroma_dir),
        backend="claude" if backend == "claude" else backend,
        api_key=args.api_key,
        model=args.model,
        collection_name=COLLECTION_NAME,
        top_k=args.top_k,
        language_filter=args.language,
    )
    response = rag.query(prompt)
    print(response.answer)
    _print_sources(response.sources)


def main(argv: list[str] | None = None) -> None:
    _configure_console()
    parser = _build_parser()
    args = parser.parse_args(argv)

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        parser.error('Enter a prompt, e.g. elder check the code')

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    index, _ = _ensure_index(args, root)
    backend = _select_backend(args.backend)

    if backend == "search":
        _local_answer(index, prompt, args)
        return

    try:
        _llm_answer(index, prompt, args, root, backend)
    except Exception as exc:
        print(f"LLM backend failed: {exc}", file=sys.stderr)
        print("\nFalling back to local code search.\n")
        _local_answer(index, prompt, args)


if __name__ == "__main__":
    main()
