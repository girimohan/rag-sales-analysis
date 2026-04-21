from typing import List


def chunk_text(text: str, max_length: int = 500) -> List[str]:
    """Split a single text into chunks of at most max_length characters.

    Uses simple character-based splitting. Returns only non-empty chunks.
    """
    chunks = []
    for i in range(0, len(text), max_length):
        chunk = text[i : i + max_length]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def chunk_text_list(texts: List[str], max_length: int = 500) -> List[str]:
    """Apply chunk_text to each entry and return a flat list of all chunks."""
    chunks = []
    for text in texts:
        chunks.extend(chunk_text(text, max_length=max_length))
    return chunks

