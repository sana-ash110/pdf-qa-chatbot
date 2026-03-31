
from __future__ import annotations

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_chunk_dict(
    text:        str,
    chunk_index: int,
    page_num:    int | None,
    source:      str,
    char_start:  int,
    char_end:    int,
) -> dict:
    page_tag = f"p{page_num}" if page_num is not None else "global"
    chunk_id = f"{source}_{page_tag}_c{chunk_index + 1}"
    return {
        "chunk_id":    chunk_id,
        "text":        text,
        "page_number": page_num,
        "source":      source,
        "char_start":  char_start,
        "char_end":    char_end,
    }


def _split_with_offsets(splitter, text: str) -> list[tuple[str, int, int]]:
    raw_chunks = splitter.split_text(text)
    results    = []
    cursor     = 0

    for chunk in raw_chunks:
        start = text.find(chunk, cursor)
        if start == -1:
            start = cursor
        end    = start + len(chunk)
        cursor = start + 1
        results.append((chunk, start, end))

    return results


# ── Strategy 1: Recursive (default) ──────────────────────────────────────────

def chunk_pages(
    pages:      list[dict],
    chunk_size: int = 500,
    overlap:    int = 100,
) -> list[dict]:
  
    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = chunk_size,
        chunk_overlap   = overlap,
        length_function = len,
        separators      = ["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[dict] = []

    for page in pages:
        text     = page["text"]
        page_num = page["page_number"]
        source   = page["source"]

        if not text.strip():
            continue

        for i, (chunk_text, start, end) in enumerate(
            _split_with_offsets(splitter, text)
        ):
            if chunk_text.strip():
                all_chunks.append(
                    _make_chunk_dict(chunk_text, i, page_num, source, start, end)
                )

    return all_chunks


# ── Strategy 2: Sentence-based ────────────────────────────────────────────────

def chunk_pages_by_sentence(
    pages:               list[dict],
    chunk_size:          int       = 500,
    overlap:             int       = 50,
    sentences_per_chunk: int | None = None,
) -> list[dict]:
    """
    Splits on sentence boundaries using NLTK.
    Requires: pip install nltk
    """
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download("punkt_tab", quiet=True)
    except ImportError:
        raise ImportError("Run: pip install nltk")

    all_chunks: list[dict] = []

    for page in pages:
        text     = page["text"]
        page_num = page["page_number"]
        source   = page["source"]

        if not text.strip():
            continue

        sentences = nltk.sent_tokenize(text)

        if sentences_per_chunk:
            step   = max(1, sentences_per_chunk - 1)
            groups = [
                " ".join(sentences[i : i + sentences_per_chunk])
                for i in range(0, len(sentences), step)
            ]
        else:
            groups      = []
            current     = []
            current_len = 0

            for sent in sentences:
                if current_len + len(sent) > chunk_size and current:
                    groups.append(" ".join(current))
                    current     = current[-1:] if overlap > 0 else []
                    current_len = sum(len(s) for s in current)
                current.append(sent)
                current_len += len(sent)

            if current:
                groups.append(" ".join(current))

        cursor = 0
        for i, chunk_text in enumerate(groups):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            start  = text.find(chunk_text[:40], cursor)
            start  = max(0, start)
            end    = start + len(chunk_text)
            cursor = start + 1
            all_chunks.append(
                _make_chunk_dict(chunk_text, i, page_num, source, start, end)
            )

    return all_chunks


# ── Strategy 3: Character (legacy) ───────────────────────────────────────────

def chunk_pages_by_character(
    pages:      list[dict],
    chunk_size: int = 500,
    overlap:    int = 100,
) -> list[dict]:
    """Pure character-count sliding window. No extra dependencies."""
    splitter = CharacterTextSplitter(
        chunk_size      = chunk_size,
        chunk_overlap   = overlap,
        separator       = " ",
        length_function = len,
    )

    all_chunks: list[dict] = []

    for page in pages:
        text     = page["text"]
        page_num = page["page_number"]
        source   = page["source"]

        if not text.strip():
            continue

        for i, (chunk_text, start, end) in enumerate(
            _split_with_offsets(splitter, text)
        ):
            if chunk_text.strip():
                all_chunks.append(
                    _make_chunk_dict(chunk_text, i, page_num, source, start, end)
                )

    return all_chunks


# ── Strategy 4: Whole-document ────────────────────────────────────────────────

def chunk_document(
    pages:      list[dict],
    chunk_size: int = 500,
    overlap:    int = 100,
) -> list[dict]:
    """Treats the entire document as a single chunk. No splitting."""
    if not pages:
        return []

    full_text = "\n\n".join(p["text"] for p in pages if p["text"].strip())
    source    = pages[0]["source"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = chunk_size,
        chunk_overlap = overlap,
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )

    return [
        _make_chunk_dict(chunk_text, i, None, source, start, end)
        for i, (chunk_text, start, end) in enumerate(
            _split_with_offsets(splitter, full_text)
        )
        if chunk_text.strip()
    ]


# ── Utility ───────────────────────────────────────────────────────────────────

def print_chunk_stats(chunks: list[dict]) -> None:
    if not chunks:
        print("No chunks.")
        return
    lengths = [len(c["text"]) for c in chunks]
    print(f"Total chunks : {len(chunks)}")
    print(f"Avg length   : {sum(lengths) // len(lengths)} chars")
    print(f"Min / Max    : {min(lengths)} / {max(lengths)} chars")
