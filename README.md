# Elder

A tree-sitter powered code indexer with a full RAG pipeline. Scans a project
directory, extracts symbols, chunks large files into ≤ 400-token slices, saves
a searchable vector index in ChromaDB, and lets you query your codebase in
plain English using Claude, OpenAI, or a local Ollama model — no cloud
required.

---

## Project layout

```
code_indexer/
├── scanner.py       — directory walker & file discovery
├── parser.py        — tree-sitter parsing + symbol extraction
├── chunker.py       — 400-token chunking with symbol-boundary awareness
├── index.py         — index assembly, JSON save/load
├── query.py         — BM25-style keyword search over the index
├── vectorstore.py   — ChromaDB persistence & semantic vector search
├── rag.py           — RAG pipeline: ChromaDB → LLM → answer
├── main.py          — CLI entry point (index, search, chroma-save, chroma-search)
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Tree-sitter is optional.** The indexer falls back to Python's built-in
> `ast` module for `.py` files and a regex-based fallback for all other
> languages. Install `tree-sitter` and language grammar packages to unlock
> full multi-language support.

---

## Full pipeline — from code to answers

```
your code files
      │
      │  Step 1 — python main.py chroma-save
      ▼
scanner.py    →  finds source files
parser.py     →  extracts symbols (functions, classes, imports …)
chunker.py    →  splits into ≤ 400-token chunks
vectorstore.py → embeds + saves to ChromaDB  (runs once)
      │
      │  Step 2 — python rag.py --prompt "..."
      ▼
vectorstore.py → semantic search: finds relevant chunks
rag.py         → builds structured prompt with retrieved code
      │          sends to Claude / OpenAI / Ollama
      ▼
    answer  +  source references
```

Step 1 is a **one-time setup**. Re-run it only when your codebase changes.

---

## Quick start

### Step 1 — Build the index and save to ChromaDB

```bash
python main.py chroma-save ./my_project --db ./my_project.chroma --verbose
```

With a prompt pre-filter (only indexes files relevant to your topic):

```bash
python main.py chroma-save ./my_project \
    --db ./my_project.chroma \
    --prompt "authentication JWT login" \
    --verbose
```

From an existing JSON index (skip re-scanning):

```bash
python main.py chroma-save --db ./my_project.chroma --from-json ./my_project.index.json
```

### Step 2 — Query with the RAG pipeline

```bash
# Using Claude
python rag.py --db ./my_project.chroma \
              --backend claude \
              --api-key sk-ant-... \
              --prompt "How does JWT authentication work in this codebase?"

# Using OpenAI
python rag.py --db ./my_project.chroma \
              --backend openai \
              --api-key sk-... \
              --prompt "Where are database connections created and closed?"

# Using Ollama locally — free, no API key needed
ollama pull deepseek-r1:14b
python rag.py --db ./my_project.chroma \
              --backend ollama \
              --model deepseek-r1:14b \
              --prompt "Why does line 42 throw a KeyError?"
```

### Interactive mode — keep asking questions

```bash
python rag.py --db ./my_project.chroma --backend ollama --model deepseek-r1:14b --interactive

You › How does user login work?
You › Where are tokens validated?
You › sources        ← shows files retrieved in the last query
You › exit
```

---

## LLM backends

| Backend | Command | API Key | Cost |
|---|---|---|---|
| `claude` | `--backend claude` | `ANTHROPIC_API_KEY` | Paid |
| `openai` | `--backend openai` | `OPENAI_API_KEY` | Paid |
| `ollama` | `--backend ollama` | None | **Free** |

### Recommended Ollama models for code

```bash
ollama pull deepseek-r1:7b      # ~4.7 GB — works on 8 GB RAM
ollama pull deepseek-r1:14b     # ~9 GB   — best balance, needs 16 GB RAM
ollama pull qwen2.5-coder:14b   # ~9 GB   — purpose-built for code
ollama pull codellama:34b       # ~20 GB  — needs 32 GB RAM
```

`deepseek-r1` uses step-by-step reasoning before answering, making it
noticeably better at debugging than plain chat models.

---

## CLI reference

### `main.py` — indexing

```bash
# Build JSON index
python main.py index ./my_project --output ./my_project.index.json --verbose

# Build + save directly to ChromaDB
python main.py chroma-save ./my_project --db ./my_project.chroma

# Push existing JSON index into ChromaDB
python main.py chroma-save --db ./my_project.chroma --from-json ./my_project.index.json

# BM25 keyword search over a JSON index
python main.py search ./my_project.index.json "database connection pooling" --top-k 5

# Semantic search directly in ChromaDB
python main.py chroma-search ./my_project.chroma "JWT token verification" --language python
```

### `rag.py` — RAG pipeline

```bash
python rag.py --db <chroma_dir>
              --backend claude|openai|ollama
              --api-key <key>          # not needed for ollama
              --model <model_name>     # e.g. deepseek-r1:14b
              --prompt "<question>"    # single query
              --interactive            # REPL mode
              --top-k 8               # chunks to retrieve (default: 8)
              --language python        # filter by language
              --show-prompt            # print the full prompt sent to the LLM
```

---

## Python API

### Build and save to ChromaDB

```python
from scanner      import scan_directory
from index        import build_index
from vectorstore  import save_to_chroma

sources = scan_directory("./my_project")
idx     = build_index("./my_project", sources, max_tokens_per_chunk=400, verbose=True)
save_to_chroma(idx, persist_dir="./my_project.chroma", verbose=True)
```

### RAG query

```python
from rag import CodeRAG

rag = CodeRAG(
    db_path = "./my_project.chroma",
    backend = "ollama",
    model   = "deepseek-r1:14b",
    top_k   = 10,
)

response = rag.query("How does the connection pool handle failures?")

print(response.answer)           # LLM's analysis in Markdown
print(response.chunks_found)     # number of chunks retrieved
for src in response.sources:
    print(src.file, src.lines, src.similarity)
```

### BM25 keyword search (no LLM)

```python
from index import load_index
from query import search, format_results

idx     = load_index("./my_project.index.json")
results = search(idx, "JWT token validation", top_k=10)
print(format_results(results))
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
          "parent": ""
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

## How search works

### Keyword search (`query.py`) — no embeddings needed
1. **TF×IDF** — term frequency inside chunk × log(N/df) across all chunks.
2. **Symbol-name exact match** — ×1.5 bonus if a keyword matches a symbol name.
3. **File-path match** — ×1.3 bonus if a keyword appears in the file path.

### Semantic search (`vectorstore.py`) — ChromaDB + embeddings
Chunk text is embedded using `all-MiniLM-L6-v2` (via ChromaDB's default
embedding function) and stored in an HNSW index. Queries are embedded the same
way and ranked by cosine similarity. Pass a custom `embedding_function` to use
OpenAI, Voyage, or any other provider.

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
