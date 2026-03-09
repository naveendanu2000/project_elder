# code-indexer

A tree-sitter powered code indexer that scans a project directory, extracts
symbols, chunks large files into ≤ 400-token slices, and saves a searchable
JSON index — all without needing a vector database.

---

## Project layout

```
code_indexer/
├── scanner.py      — directory walker & file discovery
├── parser.py       — tree-sitter parsing + symbol extraction
├── chunker.py      — 400-token chunking with symbol-boundary awareness
├── index.py        — index assembly, JSON save/load
├── query.py        — BM25-style keyword search over the index
├── main.py         — CLI entry point
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note** – The indexer works out-of-the-box **without** tree-sitter
> installed.  It falls back to Python's built-in `ast` module for `.py`
> files and a regex-based fallback for other languages.  Install
> `tree-sitter` and the language grammar packages to unlock full
> multi-language support.

---

## Quick start

### 1. Build an index

```bash
python main.py index /path/to/my_project
# → saves  /path/to/my_project.index.json
```

With options:

```bash
python main.py index ./my_project \
    --output ./indexes/myproject.json \
    --max-tokens 400 \
    --verbose
```

Pre-filter to only index files relevant to a prompt (useful for large repos):

```bash
python main.py index ./my_project --prompt "authentication and JWT"
```

### 2. Search the index

```bash
python main.py search ./my_project.index.json "how does user authentication work"
```

With filters:

```bash
python main.py search ./my_project.index.json "database connection pooling" \
    --language python \
    --top-k 5
```

---

## Python API

```python
from scanner import scan_directory
from index   import build_index, save_index, load_index
from query   import search, format_results

# --- Build ---
sources = scan_directory("./my_project")
idx     = build_index("./my_project", sources, max_tokens_per_chunk=400, verbose=True)
save_index(idx, "my_project.index.json")

# --- Search ---
idx     = load_index("my_project.index.json")
results = search(idx, "JWT token validation", top_k=10)
print(format_results(results))

# --- Use results ---
for r in results:
    print(r.chunk.rel_path, r.chunk.start_line, "→", r.chunk.end_line)
    print(r.chunk.text[:200])
```

---

## Index JSON schema

```jsonc
{
  "version": "1.0",
  "root": "/abs/path/to/project",
  "created_at": "2024-01-15T12:00:00+00:00",
  "stats": { "files": 42, "symbols": 310, "chunks": 128 },
  "files": [
    {
      "rel_path": "src/auth.py",
      "language": "python",
      "size_bytes": 3200,
      "parser_used": "treesitter",   // or "ast" or "regex"
      "parse_error": "",
      "symbols": [
        {
          "name": "verify_token",
          "kind": "function",        // function|class|method|import|…
          "start_line": 24,
          "end_line": 48,
          "signature": "def verify_token(token: str) -> dict:",
          "docstring": "Verify a JWT and return its payload.",
          "parent": ""               // or enclosing class name
        }
      ],
      "chunks": [
        {
          "chunk_id": "src/auth.py#chunk1",
          "start_line": 1,
          "end_line": 55,
          "token_count": 387,
          "symbol_names": ["verify_token", "TokenError"],
          "chunk_index": 0,
          "total_chunks": 2,
          "text": "..."
        }
      ]
    }
  ]
}
```

---

## How ranking works

The `query.search()` function uses a **BM25-inspired keyword scorer**:

1. **TF×IDF** — term frequency inside the chunk × log(N/df) across all chunks.
2. **Symbol-name exact match** — ×1.5 bonus if any keyword matches a symbol name.
3. **File-path match** — ×1.3 bonus if any keyword appears in the file path.

For production workloads, drop in any embedding model + vector store — the
`Chunk` dataclass gives you everything you need (`text`, `symbol_names`, etc.)
to generate and store embeddings without changing the rest of the pipeline.

---

## Supported languages

| Language   | Tree-sitter grammar         | Fallback     |
|------------|-----------------------------|--------------|
| Python     | `tree-sitter-python`        | `ast` module |
| JavaScript | `tree-sitter-javascript`    | regex        |
| TypeScript | `tree-sitter-typescript`    | regex        |
| C          | `tree-sitter-c`             | regex        |
| C++        | `tree-sitter-cpp`           | regex        |
| Rust       | `tree-sitter-rust`          | regex        |
| Go         | `tree-sitter-go`            | regex        |
| Java       | `tree-sitter-java`          | regex        |
| Ruby       | `tree-sitter-ruby`          | regex        |
