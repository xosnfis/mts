"""
file_processor.py — Multi-format document parser for GPTHub RAG pipeline.

Supports: .pdf, .docx, .txt, .csv, .xlsx
Extracts text and converts tabular data to Markdown for LLM consumption.
"""

import io
import logging
import base64
from typing import Optional

log = logging.getLogger(__name__)

# Token budget: ~4 chars per token, leave room for system prompt + conversation
MAX_CHARS = 12_000


def _truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    """Smart truncation: cut at paragraph boundary if possible."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_para = cut.rfind("\n\n")
    if last_para > max_chars * 0.7:
        cut = cut[:last_para]
    return cut + "\n\n[... документ обрезан из-за ограничения токенов ...]"


def _parse_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def _parse_docx(data: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(data))
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())
    return "\n\n".join(parts)


def _parse_txt(data: bytes) -> str:
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _parse_csv(data: bytes) -> str:
    import pandas as pd
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            break
        except Exception:
            continue
    else:
        return "[Ошибка чтения CSV]"
    return _df_to_markdown(df)


def _parse_xlsx(data: bytes) -> str:
    import pandas as pd
    try:
        xl = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
        parts = []
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            parts.append(f"### Лист: {sheet}\n\n{_df_to_markdown(df)}")
        return "\n\n".join(parts)
    except Exception as e:
        log.error("XLSX parse error: %s", e)
        return "[Ошибка чтения XLSX]"


def _df_to_markdown(df) -> str:
    df = df.fillna("").astype(str)
    if len(df) > 200:
        df = df.head(200)
        truncated = True
    else:
        truncated = False
    md = df.to_markdown(index=False)
    if truncated:
        md += "\n\n_[таблица обрезана до 200 строк]_"
    return md


def extract_text_from_any(filename: str, data: bytes) -> Optional[str]:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    try:
        if ext == "pdf":
            text = _parse_pdf(data)
        elif ext == "docx":
            text = _parse_docx(data)
        elif ext == "txt":
            text = _parse_txt(data)
        elif ext == "csv":
            text = _parse_csv(data)
        elif ext in ("xlsx", "xls"):
            text = _parse_xlsx(data)
        else:
            log.warning("Unsupported file extension: %s", ext)
            return None
    except Exception as e:
        log.error("File parse error (%s): %s", filename, e)
        return None

    text = text.strip()
    if not text:
        return None
    return _truncate(text)


def build_file_context_block(filename: str, text: str) -> str:
    return (
        f"[FILE CONTEXT: {filename}]\n"
        "Используй следующее содержимое файла при ответе на вопрос пользователя.\n\n"
        f"{text}"
    )
