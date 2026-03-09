"""
main.py
-------
Command-line interface for the code indexer.

Usage examples
--------------
# Build an index → JSON
python main.py index ./my_project
python main.py index ./my_project --output custom.json --max-tokens 400 --verbose

# Search a JSON index (BM25 keyword search)
python main.py search ./my_project.index.json "authentication and JWT handling"
python main.py search ./my_project.index.json "database query" --language python --top-k 5 --no-text

# Build an index AND save straight to ChromaDB (skip JSON)
python main.py chroma-save ./my_project --db ./my_project.chroma

# Save an existing JSON index into ChromaDB
python main.py chroma-save ./my_project --db ./my_project.chroma --from-json ./my_project.index.json

# Semantic search against ChromaDB
python main.py chroma-search ./my_project.chroma "how does authentication work"
python main.py chroma-search ./my_project.chroma "database pool" --language python --top-k 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_index(args: argparse.Namespace) -> None:
    from scanner import scan_directory
    from index import build_index, save_index

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else root.parent / f"{root.name}.index.json"

    # optional prompt-based pre-filter during scan
    prompt_keywords: list[str] | None = None
    if args.prompt:
        from query import _extract_keywords
        prompt_keywords = _extract_keywords(args.prompt)
        if args.verbose:
            print(f"Pre-filter keywords: {prompt_keywords}")

    print(f"Scanning  {root} …")
    t0 = time.perf_counter()
    sources = scan_directory(
        root,
        max_file_size_kb=args.max_file_size_kb,
        prompt_keywords=prompt_keywords,
    )
    print(f"  Found {len(sources)} files  ({time.perf_counter() - t0:.1f}s)")

    if not sources:
        print("No source files found. Check the root path and filters.")
        sys.exit(0)

    print("Parsing & indexing …")
    t1 = time.perf_counter()
    index = build_index(
        root=root,
        source_files=sources,
        max_tokens_per_chunk=args.max_tokens,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - t1
    stats = index.stats
    print(
        f"  {stats['files']} files | "
        f"{stats['symbols']} symbols | "
        f"{stats['chunks']} chunks  ({elapsed:.1f}s)"
    )

    saved = save_index(index, output)
    print(f"\nIndex saved → {saved}")


def cmd_search(args: argparse.Namespace) -> None:
    from index import load_index
    from query import search, format_results

    index_path = Path(args.index_file)
    if not index_path.is_file():
        print(f"ERROR: index file not found: {index_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading index from {index_path} …")
    index = load_index(index_path)
    print(f"  {index.stats['files']} files | "
          f"{index.stats['symbols']} symbols | "
          f"{index.stats['chunks']} chunks\n")

    results = search(
        index,
        args.prompt,
        top_k=args.top_k,
        min_score=args.min_score,
        language_filter=args.language or None,
    )

    print(f"Results for: \"{args.prompt}\"\n{'=' * 60}")
    print(format_results(results, show_text=not args.no_text))


def cmd_chroma_save(args: argparse.Namespace) -> None:
    from vectorstore import save_to_chroma
    from index import load_index, build_index
    from scanner import scan_directory

    db_path = Path(args.db)

    if args.from_json:
        # Load an already-built JSON index and push it to Chroma
        json_path = Path(args.from_json)
        if not json_path.is_file():
            print(f"ERROR: JSON index not found: {json_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading JSON index from {json_path} …")
        index = load_index(json_path)
    else:
        # Scan + build fresh, then push to Chroma
        root = Path(args.root).resolve()
        if not root.is_dir():
            print(f"ERROR: {root} is not a directory", file=sys.stderr)
            sys.exit(1)

        prompt_keywords = None
        if args.prompt:
            from query import _extract_keywords
            prompt_keywords = _extract_keywords(args.prompt)

        print(f"Scanning {root} …")
        sources = scan_directory(root, max_file_size_kb=args.max_file_size_kb,
                                 prompt_keywords=prompt_keywords)
        print(f"  Found {len(sources)} files")

        print("Parsing & indexing …")
        index = build_index(root, sources, max_tokens_per_chunk=args.max_tokens,
                            verbose=args.verbose)

    print(f"Saving to ChromaDB at '{db_path}' …")
    save_to_chroma(
        index,
        persist_dir=db_path,
        collection_name=args.collection,
        verbose=True,
    )


def cmd_chroma_search(args: argparse.Namespace) -> None:
    from vectorstore import query_chroma
    from query import format_results

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: ChromaDB directory not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Searching ChromaDB at '{db_path}' …\n")
    results = query_chroma(
        persist_dir=db_path,
        prompt=args.prompt,
        collection_name=args.collection,
        top_k=args.top_k,
        language_filter=args.language or None,
    )

    print(f'Results for: "{args.prompt}"\n{"=" * 60}')
    print(format_results(results, show_text=not args.no_text))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code-indexer",
        description="Tree-sitter powered code indexer for Python (and more) projects.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- index ---
    p_index = sub.add_parser("index", help="Scan a directory and build an index.")
    p_index.add_argument("root", help="Root directory to scan.")
    p_index.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON path (default: <root>.index.json).",
    )
    p_index.add_argument(
        "--max-tokens",
        type=int, default=400,
        help="Max tokens per chunk (default: 400).",
    )
    p_index.add_argument(
        "--max-file-size-kb",
        type=int, default=512,
        help="Skip files larger than this many KB (default: 512).",
    )
    p_index.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt to pre-filter which files are indexed.",
    )
    p_index.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-file progress.",
    )
    p_index.set_defaults(func=cmd_index)

    # --- search ---
    p_search = sub.add_parser("search", help="Query an existing index.")
    p_search.add_argument("index_file", help="Path to the .index.json file.")
    p_search.add_argument("prompt", help="Natural-language search query.")
    p_search.add_argument(
        "--top-k", "-k",
        type=int, default=10,
        help="Number of results to return (default: 10).",
    )
    p_search.add_argument(
        "--min-score",
        type=float, default=0.01,
        help="Minimum relevance score (default: 0.01).",
    )
    p_search.add_argument(
        "--language",
        default=None,
        help="Filter results to a specific language (e.g. python, javascript).",
    )
    p_search.add_argument(
        "--no-text",
        action="store_true",
        help="Suppress chunk source text in output.",
    )
    p_search.set_defaults(func=cmd_search)

    # --- chroma-save ---
    p_csave = sub.add_parser(
        "chroma-save",
        help="Build (or load) an index and upsert it into ChromaDB.",
    )
    p_csave.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Root directory to scan (not needed if --from-json is given).",
    )
    p_csave.add_argument(
        "--db", "-d",
        required=True,
        help="ChromaDB persistence directory (created if absent).",
    )
    p_csave.add_argument(
        "--from-json",
        default=None,
        metavar="JSON_PATH",
        help="Load an existing .index.json instead of re-scanning.",
    )
    p_csave.add_argument("--collection", default="code_index",
                         help="Chroma collection base name (default: code_index).")
    p_csave.add_argument("--max-tokens", type=int, default=400)
    p_csave.add_argument("--max-file-size-kb", type=int, default=512)
    p_csave.add_argument("--prompt", default=None,
                         help="Optional prompt to pre-filter scanned files.")
    p_csave.add_argument("--verbose", "-v", action="store_true")
    p_csave.set_defaults(func=cmd_chroma_save)

    # --- chroma-search ---
    p_csearch = sub.add_parser(
        "chroma-search",
        help="Semantic similarity search over a ChromaDB index.",
    )
    p_csearch.add_argument("db", help="ChromaDB persistence directory.")
    p_csearch.add_argument("prompt", help="Natural-language search query.")
    p_csearch.add_argument("--collection", default="code_index")
    p_csearch.add_argument("--top-k", "-k", type=int, default=10)
    p_csearch.add_argument("--language", default=None)
    p_csearch.add_argument("--no-text", action="store_true",
                           help="Suppress chunk source text in output.")
    p_csearch.set_defaults(func=cmd_chroma_search)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
