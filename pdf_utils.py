"""
PDF utilities: extract text and simple metadata using PyMuPDF (fitz).
"""
import fitz
from typing import Dict, Any

def extract_text_from_pdf(path: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "full_text": "<concatenated text>",
        "pages": [ {"page_number": int, "text": "<text>", "start_char": int, "end_char": int}, ... ],
      }
    """
    doc = fitz.open(path)
    full_text_parts = []
    pages = []
    char_cursor = 0
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        # normalize newlines
        text = text.replace('\r\n', '\n')
        start = char_cursor
        full_text_parts.append(text + "\n")
        char_cursor += len(text) + 1
        pages.append({"page_number": i+1, "text": text, "start_char": start, "end_char": char_cursor-1})
    full_text = "".join(full_text_parts)
    return {"full_text": full_text, "pages": pages}
