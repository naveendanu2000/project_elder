"""
parser.py
---------
Parses source files with tree-sitter and extracts code symbols
(functions, classes, methods, imports, variables …).

If tree-sitter or a language grammar is not installed the module falls back
to a pure-Python AST parser for Python files, and a regex-based fallback for
all other languages.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scanner import SourceFile


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    """A named code entity extracted from a source file."""
    name: str
    kind: str            # "function" | "class" | "method" | "import" | "variable" | "interface" …
    start_line: int      # 1-based
    end_line: int        # 1-based (inclusive)
    signature: str       # first line / declaration
    docstring: str       # first docstring/comment if found, else ""
    parent: str          # enclosing class name, or "" for top-level
    source_snippet: str  # raw text of the symbol body


@dataclass
class ParsedFile:
    """Result of parsing a single source file."""
    rel_path: str
    language: str
    symbols: list[Symbol] = field(default_factory=list)
    parse_error: str = ""
    parser_used: str = ""   # "treesitter" | "ast" | "regex"


# ---------------------------------------------------------------------------
# Tree-sitter bootstrap (lazy import so the package is optional)
# ---------------------------------------------------------------------------

_TS_AVAILABLE = False
_TS_PARSERS: dict[str, Any] = {}   # language -> tree_sitter.Parser


def _try_load_treesitter() -> bool:
    """Attempt to import tree_sitter and pre-load language grammars."""
    global _TS_AVAILABLE, _TS_PARSERS
    if _TS_AVAILABLE:
        return True
    try:
        from tree_sitter import Language, Parser  # type: ignore

        # tree-sitter >= 0.21 ships grammars as separate pip packages
        # e.g. tree-sitter-python, tree-sitter-javascript …
        _lang_modules = {
            "python":     ("tree_sitter_python", "language"),
            "javascript": ("tree_sitter_javascript", "language"),
            "typescript": ("tree_sitter_typescript", "language_typescript"),
            "tsx":        ("tree_sitter_typescript", "language_tsx"),
            "c":          ("tree_sitter_c", "language"),
            "cpp":        ("tree_sitter_cpp", "language"),
            "rust":       ("tree_sitter_rust", "language"),
            "go":         ("tree_sitter_go", "language"),
            "java":       ("tree_sitter_java", "language"),
            "ruby":       ("tree_sitter_ruby", "language"),
        }

        for lang_name, (module_name, language_func) in _lang_modules.items():
            try:
                import importlib
                mod = importlib.import_module(module_name)
                lang = Language(getattr(mod, language_func)())
                p = Parser(lang)
                _TS_PARSERS[lang_name] = p
            except Exception:
                pass  # grammar not installed — will fall back for this lang

        _TS_AVAILABLE = bool(_TS_PARSERS)
        return _TS_AVAILABLE
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(src: SourceFile) -> ParsedFile:
    """Parse *src* and return extracted symbols."""
    _try_load_treesitter()

    result = ParsedFile(rel_path=src.rel_path, language=src.language)

    if src.language == "unknown":
        result.parse_error = "Unsupported language"
        return result

    # --- tree-sitter path ---
    parser_lang = "tsx" if src.path.suffix.lower() == ".tsx" else src.language

    if _TS_AVAILABLE and parser_lang in _TS_PARSERS:
        try:
            result.symbols = _parse_with_treesitter(src, parser_lang)
            result.parser_used = "treesitter"
            return result
        except Exception as exc:
            result.parse_error = f"tree-sitter error: {exc}"
            # fall through to fallback

    # --- Python AST fallback ---
    if src.language == "python":
        try:
            result.symbols = _parse_python_ast(src)
            result.parser_used = "ast"
            return result
        except SyntaxError as exc:
            result.parse_error = f"SyntaxError: {exc}"
            return result

    # --- Generic regex fallback for other languages ---
    result.symbols = _parse_regex_fallback(src)
    result.parser_used = "regex"
    return result


# ---------------------------------------------------------------------------
# Tree-sitter extraction
# ---------------------------------------------------------------------------

# Node types that represent named symbols per language family
_TS_SYMBOL_QUERIES: dict[str, list[tuple[str, str]]] = {
    # (node_type, symbol_kind)
    "python": [
        ("function_definition", "function"),
        ("async_function_definition", "function"),
        ("class_definition", "class"),
        ("import_statement", "import"),
        ("import_from_statement", "import"),
    ],
    "javascript": [
        ("function_declaration", "function"),
        ("arrow_function", "function"),
        ("class_declaration", "class"),
        ("method_definition", "method"),
        ("import_statement", "import"),
        ("lexical_declaration", "variable"),
        ("variable_declaration", "variable"),
    ],
    "typescript": [
        ("function_declaration", "function"),
        ("arrow_function", "function"),
        ("class_declaration", "class"),
        ("interface_declaration", "interface"),
        ("method_definition", "method"),
        ("import_statement", "import"),
        ("type_alias_declaration", "type"),
    ],
    "tsx": [
        ("function_declaration", "function"),
        ("arrow_function", "function"),
        ("class_declaration", "class"),
        ("interface_declaration", "interface"),
        ("method_definition", "method"),
        ("import_statement", "import"),
        ("type_alias_declaration", "type"),
    ],
    "c": [
        ("function_definition", "function"),
        ("struct_specifier", "struct"),
        ("preproc_include", "import"),
    ],
    "cpp": [
        ("function_definition", "function"),
        ("class_specifier", "class"),
        ("struct_specifier", "struct"),
        ("namespace_definition", "namespace"),
    ],
    "rust": [
        ("function_item", "function"),
        ("struct_item", "struct"),
        ("enum_item", "enum"),
        ("impl_item", "impl"),
        ("use_declaration", "import"),
    ],
    "go": [
        ("function_declaration", "function"),
        ("method_declaration", "method"),
        ("type_declaration", "type"),
        ("import_declaration", "import"),
    ],
    "java": [
        ("class_declaration", "class"),
        ("method_declaration", "method"),
        ("interface_declaration", "interface"),
        ("import_declaration", "import"),
    ],
    "ruby": [
        ("method", "method"),
        ("class", "class"),
        ("module", "module"),
    ],
}


def _parse_with_treesitter(src: SourceFile, parser_lang: str) -> list[Symbol]:
    parser = _TS_PARSERS[parser_lang]
    tree = parser.parse(src.content.encode("utf-8"))
    lines = src.content.splitlines()

    queries = _TS_SYMBOL_QUERIES.get(parser_lang, _TS_SYMBOL_QUERIES.get(src.language, []))
    symbols: list[Symbol] = []

    def _walk(node: Any, parent_name: str = "") -> None:
        for node_type, kind in queries:
            if node.type == node_type:
                name = _ts_node_name(node, src.content)
                start = node.start_point[0]   # 0-based row
                end   = node.end_point[0]     # 0-based row
                signature = lines[start] if start < len(lines) else ""
                snippet = "\n".join(lines[start : end + 1])
                docstring = _ts_docstring(node, src.content)
                symbols.append(Symbol(
                    name=name or "<anonymous>",
                    kind=kind,
                    start_line=start + 1,
                    end_line=end + 1,
                    signature=signature.strip(),
                    docstring=docstring,
                    parent=parent_name,
                    source_snippet=snippet,
                ))
                # descend with updated parent name for nested class/function
                new_parent = name or parent_name
                for child in node.children:
                    _walk(child, new_parent)
                return   # already descended

        for child in node.children:
            _walk(child, parent_name)

    _walk(tree.root_node)
    return symbols


def _ts_node_name(node: Any, source: str) -> str:
    """Extract the identifier name from a tree-sitter node."""
    source_bytes = source.encode("utf-8")

    def _node_text(n: Any) -> str:
        return source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

    named_child = node.child_by_field_name("name")
    if named_child is not None:
        return _node_text(named_child)

    if "import" in node.type:
        first_line = _node_text(node).splitlines()[0]
        return first_line.strip()

    for child in node.children:
        if child.type == "identifier" or child.type == "name":
            return _node_text(child)
    # last resort: first word on the declaration line
    text = source_bytes[node.start_byte:node.start_byte + 80].decode("utf-8", errors="replace")
    m = re.search(r"\b([A-Za-z_]\w*)\b", text)
    return m.group(1) if m else ""


def _ts_docstring(node: Any, source: str) -> str:
    """Attempt to pull the first string literal child as a docstring."""
    source_bytes = source.encode("utf-8")

    def _node_text(n: Any) -> str:
        return source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

    for child in node.children:
        if child.type in ("block", "statement_block", "body"):
            for grandchild in child.children:
                if grandchild.type in ("expression_statement",):
                    for ggc in grandchild.children:
                        if ggc.type in ("string", "string_literal"):
                            raw = _node_text(ggc)
                            return raw.strip("'\"` \n")
    return ""


# ---------------------------------------------------------------------------
# Python AST fallback
# ---------------------------------------------------------------------------

def _parse_python_ast(src: SourceFile) -> list[Symbol]:
    tree = ast.parse(src.content, filename=src.rel_path)
    lines = src.content.splitlines()
    symbols: list[Symbol] = []

    def _visit(node: ast.AST, parent: str = "") -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "function" if not parent else "method"
            doc = ast.get_docstring(node) or ""
            start = node.lineno
            end = getattr(node, "end_lineno", node.lineno)
            sig = lines[start - 1].strip() if start <= len(lines) else ""
            snippet = "\n".join(lines[start - 1: end])
            symbols.append(Symbol(
                name=node.name,
                kind=kind,
                start_line=start,
                end_line=end,
                signature=sig,
                docstring=doc,
                parent=parent,
                source_snippet=snippet,
            ))
            for child in ast.iter_child_nodes(node):
                _visit(child, parent=node.name)

        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            start = node.lineno
            end = getattr(node, "end_lineno", node.lineno)
            sig = lines[start - 1].strip() if start <= len(lines) else ""
            snippet = "\n".join(lines[start - 1: end])
            symbols.append(Symbol(
                name=node.name,
                kind="class",
                start_line=start,
                end_line=end,
                signature=sig,
                docstring=doc,
                parent=parent,
                source_snippet=snippet,
            ))
            for child in ast.iter_child_nodes(node):
                _visit(child, parent=node.name)

        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            line = getattr(node, "lineno", 0)
            sig = lines[line - 1].strip() if line <= len(lines) else ""
            symbols.append(Symbol(
                name=", ".join(names),
                kind="import",
                start_line=line,
                end_line=line,
                signature=sig,
                docstring="",
                parent=parent,
                source_snippet=sig,
            ))
        else:
            for child in ast.iter_child_nodes(node):
                _visit(child, parent)

    for node in ast.iter_child_nodes(tree):
        _visit(node)

    return symbols


# ---------------------------------------------------------------------------
# Regex fallback (non-Python, no tree-sitter grammar available)
# ---------------------------------------------------------------------------

_REGEX_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "javascript": [
        (r"(?:^|\s)(?:function\s+)(\w+)\s*\(", "function"),
        (r"(?:^|\s)class\s+(\w+)", "class"),
        (r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(", "function"),
        (r"^import\s+", "import"),
    ],
    "typescript": [
        (r"(?:function\s+)(\w+)\s*[\(<]", "function"),
        (r"(?:^|\s)class\s+(\w+)", "class"),
        (r"(?:^|\s)interface\s+(\w+)", "interface"),
        (r"^import\s+", "import"),
        (r"(?:^|\s)type\s+(\w+)\s*=", "type"),
    ],
    "rust": [
        (r"(?:pub\s+)?fn\s+(\w+)\s*[\(<]", "function"),
        (r"(?:pub\s+)?struct\s+(\w+)", "struct"),
        (r"(?:pub\s+)?enum\s+(\w+)", "enum"),
        (r"^use\s+", "import"),
    ],
    "go": [
        (r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", "function"),
        (r"^type\s+(\w+)\s+struct", "struct"),
        (r"^import\s+", "import"),
    ],
}


def _parse_regex_fallback(src: SourceFile) -> list[Symbol]:
    patterns = _REGEX_PATTERNS.get(src.language, [])
    if not patterns:
        # Generic: try to grab any function/class-like declaration
        patterns = [
            (r"(?:function|func|def|fn)\s+(\w+)\s*[\(<]", "function"),
            (r"(?:class|struct|interface)\s+(\w+)", "class"),
        ]

    lines = src.content.splitlines()
    symbols: list[Symbol] = []

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        for pattern, kind in patterns:
            m = re.search(pattern, stripped, re.MULTILINE)
            if m:
                name = m.group(1) if m.lastindex and m.lastindex >= 1 else stripped[:40]
                symbols.append(Symbol(
                    name=name,
                    kind=kind,
                    start_line=lineno,
                    end_line=lineno,
                    signature=stripped,
                    docstring="",
                    parent="",
                    source_snippet=stripped,
                ))
                break  # one symbol per line

    return symbols
