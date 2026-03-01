"""Utility functions for the RAG-based academic notes assistant."""


def chunk_text(long_string: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split a long string into overlapping chunks.

    Args:
        long_string: The text to split.
        chunk_size: Maximum characters per chunk. Default 500.
        overlap: Number of characters to overlap between consecutive chunks. Default 100.

    Returns:
        List of text chunks.
    """
    if not long_string:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(long_string):
        end = start + chunk_size
        chunks.append(long_string[start:end])
        start += step
        if end >= len(long_string):
            break

    return chunks


def load_text_file(file_path: str) -> str:
    """
    Read a text file and return its content.

    Args:
        file_path: Path to the text file.

    Returns:
        File contents as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: On read errors.
    """
    with open(file_path, encoding="utf-8") as f:
        return f.read()
