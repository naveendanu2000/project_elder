"""
rag.py
------
RAG (Retrieval-Augmented Generation) pipeline for code analysis.

Flow
----
User prompt
    │
    ▼
ChromaDB vector search          ← vectorstore.query_chroma()
    │  top-k relevant chunks
    ▼
Context builder                 ← _build_context()
    │  structured prompt with retrieved code
    ▼
LLM (Claude / OpenAI / Ollama)  ← LLMBackend implementations
    │
    ▼
RAGResponse (answer + sources)

Supported LLM backends
----------------------
- "claude"   → Anthropic Claude  (pip install anthropic)
- "openai"   → OpenAI GPT        (pip install openai)
- "ollama"   → local Ollama      (pip install ollama)

Usage
-----
# Quickstart
from rag import CodeRAG

rag = CodeRAG(db_path="./my_project.chroma", backend="claude", api_key="sk-...")
response = rag.query("How does JWT authentication work in this codebase?")
print(response.answer)
for src in response.sources:
    print(src)

# CLI
python rag.py --db ./my_project.chroma --backend claude --api-key sk-... \\
              --prompt "How does authentication work?"
"""

from __future__ import annotations

import os
import sys
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from vectorstore import query_chroma
from query import SearchResult


def _configure_console() -> None:
    """Prefer UTF-8 output on Windows consoles that default to cp1252."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_console()


# ── constants ─────────────────────────────────────────────────────────────────

MAX_CONTEXT_CHUNKS   = 8      # how many retrieved chunks to pass to the LLM
MAX_CHUNK_CHARS      = 1500   # truncate individual chunks to avoid huge prompts
MAX_CONTEXT_CHARS    = 12_000 # hard ceiling on total context sent to LLM


# ─────────────────────────────────────────────────────────────────────────────
# Response dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceReference:
    """A single retrieved chunk that contributed to the answer."""
    file:         str
    lines:        str          # e.g. "24-48"
    symbols:      list[str]
    similarity:   float
    snippet:      str          # first few lines of the chunk


@dataclass
class RAGResponse:
    """Full result of a RAG query."""
    answer:       str
    sources:      list[SourceReference]
    prompt_used:  str = field(repr=False)   # the exact prompt sent to the LLM
    backend:      str = ""
    model:        str = ""
    chunks_found: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# LLM backend abstraction
# ─────────────────────────────────────────────────────────────────────────────

class LLMBackend(ABC):
    """Abstract base — implement one method to add a new LLM provider."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send prompts to the LLM and return the response text."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class ClaudeBackend(LLMBackend):
    """Anthropic Claude via the official Python SDK."""

    def __init__(self, api_key: str, model: str = "claude-opus-4-5"):
        try:
            import anthropic  # type: ignore
        except ImportError:
            raise ImportError("Run: pip install anthropic")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model  = model

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        message = self._client.messages.create(
            model      = self._model,
            max_tokens = 4096,
            system     = system_prompt,
            messages   = [{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text


class OpenAIBackend(LLMBackend):
    """OpenAI GPT via the official Python SDK."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            raise ImportError("Run: pip install openai")
        self._client = OpenAI(api_key=api_key)
        self._model  = model

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model    = self._model,
            messages = [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ],
            max_tokens = 4096,
        )
        return resp.choices[0].message.content


class OllamaBackend(LLMBackend):
    """Local Ollama (e.g. codellama, deepseek-coder, llama3)."""

    def __init__(self, model: str = "codellama"):
        try:
            import ollama  # type: ignore
        except ImportError:
            raise ImportError("Run: pip install ollama  (and start the Ollama daemon)")
        self._ollama = ollama
        self._model  = model

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._ollama.chat(
            model    = self._model,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return resp["message"]["content"]


def _make_backend(backend: str, api_key: str | None, model: str | None) -> LLMBackend:
    backend = backend.lower()
    if backend == "claude":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("Claude backend requires api_key or ANTHROPIC_API_KEY env var")
        return ClaudeBackend(api_key=key, model=model or "claude-opus-4-5")
    elif backend == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OpenAI backend requires api_key or OPENAI_API_KEY env var")
        return OpenAIBackend(api_key=key, model=model or "gpt-4o")
    elif backend == "ollama":
        return OllamaBackend(model=model or "codellama")
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose: claude | openai | ollama")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert code analyst assistant.
    You are given relevant code snippets retrieved from a real codebase, \
followed by a question about that code.

    Your job:
    - Answer the question thoroughly based ONLY on the provided code snippets.
    - Cite specific file names, function names, and line numbers when relevant.
    - If the retrieved snippets don't contain enough information to fully \
answer the question, say so explicitly — do not hallucinate code that isn't shown.
    - When pointing out issues or patterns, be specific and reference the actual code.
    - Format your answer in clear Markdown with code blocks where helpful.
""")


def _build_context(results: list[SearchResult], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Assemble the retrieved chunks into a structured context string for the LLM.
    """
    sections: list[str] = []
    total_chars = 0

    for i, r in enumerate(results, start=1):
        c = r.chunk
        symbols_str = ", ".join(c.symbol_names) if c.symbol_names else "—"

        # truncate individual chunk if very long
        snippet = c.text
        if len(snippet) > MAX_CHUNK_CHARS:
            snippet = snippet[:MAX_CHUNK_CHARS] + "\n... [truncated]"

        section = (
            f"### Snippet {i} — `{c.rel_path}` (lines {c.start_line}–{c.end_line})\n"
            f"**Symbols:** {symbols_str}  |  **Similarity:** {r.score:.3f}\n"
            f"```{r.file_entry.language}\n{snippet}\n```"
        )

        if total_chars + len(section) > max_chars:
            break

        sections.append(section)
        total_chars += len(section)

    return "\n\n".join(sections)


def _build_user_prompt(user_question: str, context: str) -> str:
    return textwrap.dedent(f"""\
        ## Retrieved Code Context

        {context}

        ---

        ## Question

        {user_question}
    """)


def _build_sources(results: list[SearchResult]) -> list[SourceReference]:
    sources = []
    for r in results:
        c = r.chunk
        first_lines = "\n".join(c.text.splitlines()[:6])
        sources.append(SourceReference(
            file       = c.rel_path,
            lines      = f"{c.start_line}-{c.end_line}",
            symbols    = c.symbol_names,
            similarity = round(r.score, 4),
            snippet    = first_lines,
        ))
    return sources


# ─────────────────────────────────────────────────────────────────────────────
# Main RAG class
# ─────────────────────────────────────────────────────────────────────────────

class CodeRAG:
    """
    RAG pipeline that retrieves code chunks from ChromaDB and sends them
    to an LLM for analysis.

    Parameters
    ----------
    db_path:
        ChromaDB persistence directory (created by ``chroma-save``).
    backend:
        LLM provider — ``"claude"``, ``"openai"``, or ``"ollama"``.
    api_key:
        API key for the chosen backend.  Can also be set via env vars
        ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``.
    model:
        Model name override, e.g. ``"claude-opus-4-5"``, ``"gpt-4o"``,
        ``"codellama"``.
    collection_name:
        Chroma collection base name (default: ``"code_index"``).
    top_k:
        Number of chunks to retrieve per query (default: 8).
    language_filter:
        Restrict retrieval to one language (e.g. ``"python"``).
    embedding_function:
        Custom Chroma embedding function (must match the one used at save time).
    """

    def __init__(
        self,
        db_path:            str,
        backend:            str            = "claude",
        api_key:            str | None     = None,
        model:              str | None     = None,
        collection_name:    str            = "code_index",
        top_k:              int            = MAX_CONTEXT_CHUNKS,
        language_filter:    str | None     = None,
        embedding_function: Any | None     = None,
    ):
        self.db_path            = db_path
        self.collection_name    = collection_name
        self.top_k              = top_k
        self.language_filter    = language_filter
        self.embedding_function = embedding_function
        self._llm               = _make_backend(backend, api_key, model)

    # ── public ────────────────────────────────────────────────────────────────

    def query(self, user_prompt: str) -> RAGResponse:
        """
        Full RAG query:
          1. Retrieve top-k relevant chunks from ChromaDB.
          2. Build a structured prompt with retrieved code as context.
          3. Send to the LLM.
          4. Return answer + source references.
        """
        # 1. Retrieve
        results = query_chroma(
            persist_dir        = self.db_path,
            prompt             = user_prompt,
            collection_name    = self.collection_name,
            top_k              = self.top_k,
            language_filter    = self.language_filter,
            embedding_function = self.embedding_function,
        )

        if not results:
            return RAGResponse(
                answer       = "No relevant code found in the index for your query.",
                sources      = [],
                prompt_used  = user_prompt,
                backend      = type(self._llm).__name__,
                model        = self._llm.model_name,
                chunks_found = 0,
            )

        # 2. Build prompt
        context     = _build_context(results)
        full_prompt = _build_user_prompt(user_prompt, context)

        # 3. Call LLM
        answer = self._llm.complete(
            system_prompt = SYSTEM_PROMPT,
            user_prompt   = full_prompt,
        )

        # 4. Package response
        return RAGResponse(
            answer       = answer,
            sources      = _build_sources(results),
            prompt_used  = full_prompt,
            backend      = type(self._llm).__name__,
            model        = self._llm.model_name,
            chunks_found = len(results),
        )

    def interactive(self) -> None:
        """
        Start an interactive REPL loop.
        Type 'exit' or 'quit' to stop, 'sources' to see last retrieved chunks.
        """
        print(f"\n{'═' * 60}")
        print(f"  Code RAG  │  backend={self._llm.model_name}  │  db={self.db_path}")
        print(f"{'═' * 60}")
        print("  Ask anything about the codebase. Type 'exit' to quit.\n")

        last_sources: list[SourceReference] = []

        while True:
            try:
                user_input = input("You › ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("Bye.")
                break
            if user_input.lower() == "sources":
                if not last_sources:
                    print("  (no previous query)\n")
                else:
                    print("\n── Sources from last query ──")
                    for s in last_sources:
                        syms = ", ".join(s.symbols) or "—"
                        print(f"  {s.file}:{s.lines}  [{syms}]  score={s.similarity}")
                    print()
                continue

            response = self.query(user_input)
            last_sources = response.sources

            print(f"\n{'─' * 60}")
            print(response.answer)
            print(f"\n── {response.chunks_found} chunks retrieved ──")
            for s in response.sources:
                syms = ", ".join(s.symbols) or "—"
                print(f"  {s.file}:{s.lines}  score={s.similarity}  [{syms}]")
            print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG pipeline: retrieve code from ChromaDB → analyze with LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Single query
              python rag.py --db ./my_project.chroma \\
                            --backend claude \\
                            --prompt "How does JWT authentication work?"

              # Interactive mode
              python rag.py --db ./my_project.chroma --backend openai --interactive

              # Use Ollama locally (no API key needed)
              python rag.py --db ./my_project.chroma --backend ollama --model codellama \\
                            --prompt "Explain the database connection pool"
        """),
    )
    parser.add_argument("--db",          required=True,  help="ChromaDB directory path.")
    parser.add_argument("--backend",     default="claude",
                        choices=["claude", "openai", "ollama"],
                        help="LLM backend (default: claude).")
    parser.add_argument("--api-key",     default=None,   help="API key (or set env var).")
    parser.add_argument("--model",       default=None,   help="Model name override.")
    parser.add_argument("--collection",  default="code_index")
    parser.add_argument("--top-k",       type=int, default=MAX_CONTEXT_CHUNKS,
                        help=f"Chunks to retrieve (default: {MAX_CONTEXT_CHUNKS}).")
    parser.add_argument("--language",    default=None,   help="Filter by language.")
    parser.add_argument("--prompt",      default=None,   help="Single query prompt.")
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive REPL mode.")
    parser.add_argument("--show-prompt", action="store_true",
                        help="Print the full prompt sent to the LLM.")

    args = parser.parse_args()

    if not args.prompt and not args.interactive:
        parser.error("Provide --prompt or --interactive")

    rag = CodeRAG(
        db_path         = args.db,
        backend         = args.backend,
        api_key         = args.api_key,
        model           = args.model,
        collection_name = args.collection,
        top_k           = args.top_k,
        language_filter = args.language,
    )

    if args.interactive:
        rag.interactive()
    else:
        response = rag.query(args.prompt)

        print(f"\n{'═' * 60}")
        print(f"  Model : {response.model}")
        print(f"  Chunks: {response.chunks_found} retrieved")
        print(f"{'═' * 60}\n")
        print(response.answer)
        print(f"\n── Sources ──")
        for s in response.sources:
            syms = ", ".join(s.symbols) or "—"
            print(f"  {s.file}:{s.lines}  score={s.similarity}  [{syms}]")

        if args.show_prompt:
            print(f"\n── Full prompt sent to LLM ──\n{response.prompt_used}")


if __name__ == "__main__":
    _cli()
