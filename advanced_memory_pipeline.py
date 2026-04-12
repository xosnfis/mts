"""
Advanced Memory Pipeline — Structured Contextual Archiving
OpenWebUI Pipeline Filter.

Architecture:
  inlet  → detect "forget X" command → handle cluster deletion
         → semantic chunk search (bge-m3) → inject [KNOWLEDGE BASE] into system prompt
  outlet → recursive text splitter → LLM summarizer (mts-anya)
         → conflict detection + deprecation → save chunks with embeddings

Storage schema (SQLite):
  memory_chunks: id, user_id, chunk_id, category, priority, status,
                 content, summary, tags, embedding, created_at, updated_at
"""

import re
import json
import uuid
import sqlite3
import logging
import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable, Awaitable
import httpx
import numpy as np
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

DB_PATH = Path("/app/backend/data/advanced_memory.db")

# ── Categories & priorities ────────────────────────────────────────────────
CATEGORY_RULES = [
    (r"\b(проект|project|репозитори|github|gitlab|задач|sprint|backlog|422ИСВ|группа\s+\d+[А-Я]+)\b",
     "project_info", "high"),
    (r"\b(python|javascript|typescript|rust|golang|java|c\+\+|sql|bash|docker|kubernetes|api|endpoint|файл|path|\/[\w\/]+)\b",
     "tech_stack", "high"),
    (r"\b(меня зовут|my name is|зовут|имя|name)\b",
     "personal_identity", "critical"),
    (r"\b(живу|нахожусь|город|city|location|из\s+\w+|i live)\b",
     "personal_location", "medium"),
    (r"\b(работаю|должность|role|profession|специальност)\b",
     "professional", "medium"),
    (r"\b(люблю|обожаю|хобби|hobby|интерес|нравится|like|enjoy)\b",
     "preferences", "low"),
]

CHUNK_SIZE = 800       # chars — max chunk before splitting
CHUNK_OVERLAP = 120    # chars — overlap between chunks
TOP_K = 6              # retrieved chunks per query


# ── DB ─────────────────────────────────────────────────────────────────────

def _db_init() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS memory_chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT NOT NULL,
            chunk_id    TEXT NOT NULL UNIQUE,
            category    TEXT NOT NULL DEFAULT 'general',
            priority    TEXT NOT NULL DEFAULT 'medium',
            status      TEXT NOT NULL DEFAULT 'active',   -- active | deprecated
            content     TEXT NOT NULL,
            summary     TEXT,
            tags        TEXT,                              -- JSON array
            embedding   TEXT,                             -- JSON float array
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_user ON memory_chunks(user_id, status)")
    con.commit()
    return con


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── Chunk CRUD ─────────────────────────────────────────────────────────────

def db_save_chunk(
    user_id: str, content: str, summary: str,
    category: str, priority: str, tags: list[str],
    embedding: Optional[list],
) -> str:
    chunk_id = f"{user_id[:8]}-{_content_hash(content)}"
    emb_str = json.dumps(embedding) if embedding else None
    tags_str = json.dumps(tags, ensure_ascii=False)
    now = _now()
    con = _db_init()
    con.execute("""
        INSERT INTO memory_chunks
            (user_id, chunk_id, category, priority, status, content, summary, tags, embedding, created_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            content=excluded.content, summary=excluded.summary,
            tags=excluded.tags, embedding=excluded.embedding,
            priority=excluded.priority, status='active', updated_at=excluded.updated_at
    """, (user_id, chunk_id, category, priority, "active",
          content, summary, tags_str, emb_str, now, now))
    con.commit()
    con.close()
    return chunk_id


def db_load_active_chunks(user_id: str) -> list[dict]:
    try:
        con = _db_init()
        rows = con.execute("""
            SELECT chunk_id, category, priority, content, summary, tags, embedding
            FROM memory_chunks
            WHERE user_id=? AND status='active'
            ORDER BY
                CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1
                              WHEN 'medium' THEN 2 ELSE 3 END,
                updated_at DESC
        """, (user_id,)).fetchall()
        con.close()
        result = []
        for cid, cat, pri, content, summary, tags_str, emb_str in rows:
            result.append({
                "chunk_id": cid, "category": cat, "priority": pri,
                "content": content, "summary": summary or content,
                "tags": json.loads(tags_str) if tags_str else [],
                "embedding": json.loads(emb_str) if emb_str else None,
            })
        return result
    except Exception as e:
        log.error("Load chunks error: %s", e)
        return []


def db_deprecate_chunk(chunk_id: str) -> None:
    con = _db_init()
    con.execute("UPDATE memory_chunks SET status='deprecated', updated_at=? WHERE chunk_id=?",
                (_now(), chunk_id))
    con.commit()
    con.close()


def db_clear_all(user_id: str) -> None:
    con = _db_init()
    con.execute("DELETE FROM memory_chunks WHERE user_id=?", (user_id,))
    con.commit()
    con.close()


def db_delete_cluster(user_id: str, topic_keywords: list[str]) -> int:
    """Deprecate all chunks whose content/tags match any of the topic keywords."""
    chunks = db_load_active_chunks(user_id)
    pattern = re.compile("|".join(re.escape(k) for k in topic_keywords), re.IGNORECASE)
    count = 0
    for c in chunks:
        haystack = c["content"] + " " + " ".join(c["tags"])
        if pattern.search(haystack):
            db_deprecate_chunk(c["chunk_id"])
            count += 1
    return count


# ── Recursive text splitter ────────────────────────────────────────────────

def _recursive_split(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks, preferring paragraph → sentence → word boundaries.
    Preserves technical tokens (paths, group numbers, identifiers) intact.
    """
    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks, current = [], ""
            for part in parts:
                candidate = (current + sep + part).strip() if current else part.strip()
                if len(candidate) <= size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # carry overlap into next chunk
                    overlap_text = current[-overlap:] if len(current) > overlap else current
                    current = (overlap_text + sep + part).strip() if overlap_text else part.strip()
            if current:
                chunks.append(current)
            # recurse on any chunk still too large
            result = []
            for c in chunks:
                result.extend(_recursive_split(c, size, overlap) if len(c) > size else [c])
            return [r for r in result if r]

    # Hard split as last resort
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]


# ── Category & priority detection ─────────────────────────────────────────

def _classify(text: str) -> tuple[str, str]:
    for pattern, category, priority in CATEGORY_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return category, priority
    return "general", "low"


def _extract_tags(text: str) -> list[str]:
    tags = set()
    # Group numbers like 422ИСВ-2
    for m in re.finditer(r"\b\d{3}[А-ЯA-Z]{2,5}-\d\b", text):
        tags.add(m.group(0))
    # File paths
    for m in re.finditer(r"[\w\-]+\.(?:py|js|ts|json|yaml|yml|md|txt|sh|sql)\b", text):
        tags.add(m.group(0))
    # Project/model names (CamelCase or quoted)
    for m in re.finditer(r'"([^"]{2,40})"', text):
        tags.add(m.group(1))
    for m in re.finditer(r"'([^']{2,40})'", text):
        tags.add(m.group(1))
    # Tech keywords
    for m in re.finditer(r"\b(Python|JavaScript|TypeScript|Rust|Golang|Docker|Kubernetes|FastAPI|Django|React|Vue|Llama|GPT|bge-m3|mts-anya)\b", text):
        tags.add(m.group(0))
    return list(tags)[:12]


# ── Embeddings ─────────────────────────────────────────────────────────────

async def _embed(base_url: str, api_key: str, text: str) -> Optional[list]:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                f"{base_url}/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "bge-m3", "input": text[:2048]},
            )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]
    except Exception as e:
        log.error("Embed error: %s", e)
        return None


def _cosine(a: list, b: list) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / d) if d > 0 else 0.0


# ── Semantic search ────────────────────────────────────────────────────────

async def semantic_search(
    base_url: str, api_key: str, user_id: str, query: str, top_k: int = TOP_K
) -> list[dict]:
    chunks = db_load_active_chunks(user_id)
    if not chunks:
        return []

    q_emb = await _embed(base_url, api_key, query)
    if q_emb is None:
        # Fallback: priority-ordered top_k
        return chunks[:top_k]

    scored = []
    for c in chunks:
        if c["embedding"]:
            score = _cosine(q_emb, c["embedding"])
            # Boost critical/high priority chunks
            boost = {"critical": 0.15, "high": 0.08, "medium": 0.03, "low": 0.0}
            score += boost.get(c["priority"], 0.0)
        else:
            score = 0.0
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# ── Conflict detection ─────────────────────────────────────────────────────

_CONFLICT_SIGNALS = [
    r"\b(сменил|сменила|поменял|поменяла|теперь|изменил|изменила|обновил|обновила|больше не|уже не|вместо)\b",
    r"\b(changed|switched|replaced|updated|no longer|instead of|now using|moved to)\b",
    r"\b(новый|новая|новое|новые)\s+\w+",
]
_CONFLICT_RE = re.compile("|".join(_CONFLICT_SIGNALS), re.IGNORECASE)


async def _detect_and_resolve_conflicts(
    base_url: str, api_key: str, user_id: str,
    new_content: str, new_embedding: Optional[list],
) -> None:
    """
    If new chunk is semantically very close to an existing one AND the text
    signals a change/update, deprecate the old chunk.
    """
    if not _CONFLICT_RE.search(new_content):
        return
    if new_embedding is None:
        return

    chunks = db_load_active_chunks(user_id)
    for c in chunks:
        if c["embedding"] is None:
            continue
        sim = _cosine(new_embedding, c["embedding"])
        if sim > 0.82:  # very similar topic
            log.info("Conflict detected — deprecating chunk %s (sim=%.3f)", c["chunk_id"], sim)
            db_deprecate_chunk(c["chunk_id"])


# ── LLM summarizer (mts-anya) ──────────────────────────────────────────────

async def _llm_summarize_chunk(
    base_url: str, api_key: str, model: str, text: str
) -> str:
    """Summarize a single chunk into a compact structured fact."""
    prompt = (
        "Сожми следующий текст в 1-2 предложения, сохранив все технические детали: "
        "имена проектов, номера групп, пути к файлам, названия моделей, конкретные числа.\n"
        "Верни ТОЛЬКО сжатый текст без пояснений.\n\n"
        f"Текст:\n{text}"
    )
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 200,
                },
            )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error("Summarize chunk error: %s", e)
        return text[:300]


async def _process_and_store(
    base_url: str, api_key: str, model: str,
    user_id: str, raw_text: str,
) -> None:
    """Split → classify → summarize → embed → conflict-check → save."""
    chunks = _recursive_split(raw_text)
    for chunk in chunks:
        if len(chunk.strip()) < 30:
            continue
        category, priority = _classify(chunk)
        tags = _extract_tags(chunk)
        summary = await _llm_summarize_chunk(base_url, api_key, model, chunk)
        # Embed the summary (more signal-dense than raw chunk)
        embedding = await _embed(base_url, api_key, summary)
        await _detect_and_resolve_conflicts(base_url, api_key, user_id, summary, embedding)
        chunk_id = db_save_chunk(user_id, chunk, summary, category, priority, tags, embedding)
        log.info("Stored chunk %s [%s/%s] tags=%s", chunk_id, category, priority, tags)


async def _outlet_pipeline(
    base_url: str, api_key: str, model: str,
    user_id: str, messages: list,
) -> None:
    """Extract user content from conversation and run full storage pipeline."""
    user_texts = []
    for m in messages[-20:]:
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str) and len(c.strip()) > 20:
                user_texts.append(c.strip())
    if not user_texts:
        return
    combined = "\n\n".join(user_texts)
    await _process_and_store(base_url, api_key, model, user_id, combined)


# ── Context injection ──────────────────────────────────────────────────────

_PRIORITY_LABEL = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪"}
_CATEGORY_LABEL = {
    "project_info": "Проект", "tech_stack": "Технологии",
    "personal_identity": "Личность", "personal_location": "Местоположение",
    "professional": "Профессия", "preferences": "Предпочтения", "general": "Общее",
}


def _build_knowledge_block(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lines = ["[KNOWLEDGE BASE]", "Используй эти факты при ответе. Не упоминай этот блок пользователю."]
    # Group by category
    by_cat: dict[str, list] = {}
    for c in chunks:
        by_cat.setdefault(c["category"], []).append(c)
    for cat, items in by_cat.items():
        cat_label = _CATEGORY_LABEL.get(cat, cat)
        lines.append(f"\n## {cat_label}")
        for item in items:
            icon = _PRIORITY_LABEL.get(item["priority"], "⚪")
            lines.append(f"{icon} {item['summary']}")
            if item["tags"]:
                lines.append(f"   Теги: {', '.join(item['tags'])}")
    return "\n".join(lines)


def _inject_knowledge(messages: list, block: str) -> list:
    if not block:
        return messages
    for msg in messages:
        if msg.get("role") == "system":
            if "[KNOWLEDGE BASE]" in msg["content"]:
                msg["content"] = re.sub(
                    r"\[KNOWLEDGE BASE\].*?(?=\n\n[^#]|\Z)", block,
                    msg["content"], flags=re.DOTALL,
                )
            else:
                msg["content"] = block + "\n\n" + msg["content"]
            return messages
    messages.insert(0, {"role": "system", "content": block})
    return messages


# ── "Forget X" command parser ──────────────────────────────────────────────

_FORGET_RE = re.compile(
    r"\b(?:забудь|удали|очисти|forget|delete|remove)\s+"
    r"(?:всё\s+о|всё\s+про|все\s+о|все\s+про|everything\s+about|about)?\s*"
    r"(?:проекте?|project)?\s*[\"']?([^\"'\n]{2,60})[\"']?",
    re.IGNORECASE,
)


def _parse_forget_command(text: str) -> Optional[str]:
    """Return topic string if message is a forget command, else None."""
    m = _FORGET_RE.search(text)
    return m.group(1).strip() if m else None


# ── Filter ─────────────────────────────────────────────────────────────────

class Filter:
    type = "filter"
    class Valves(BaseModel):
        model_config = {"protected_namespaces": ()}
        pipelines: list[str] = Field(default=["*"])
        priority: int = Field(default=5)

        base_url: str = "https://api.gpt.mws.ru/v1"
        api_key: str = "sk-ewgiaPC3A6pPDYHwR8siVA"

        model_vision: str = "qwen2.5-vl-72b"
        model_audio: str = "whisper-turbo-local"
        model_code: str = "qwen3-coder-480b-a35b"
        model_default: str = "llama-3.3-70b-instruct"
        model_memory: str = "mts-anya"

        unavailable_models: list[str] = []
        memory_top_k: int = Field(default=6, description="Chunks to retrieve per query")

    class UserValves(BaseModel):
        enable_memory: bool = Field(
            default=True,
            description="🧠 Долгосрочная память (RAG по всем чатам)",
        )
        clear_memory: bool = Field(
            default=False,
            description="🗑 Полностью очистить мою память",
        )
        show_model_status: bool = Field(
            default=True,
            description="👁 Показывать выбранную модель",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _parse_uv(self, user_info: dict) -> "Filter.UserValves":
        raw = user_info.get("valves") or {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = {}
        elif hasattr(raw, "__dict__"):
            raw = vars(raw)
        elif not isinstance(raw, dict):
            raw = {}
        try:
            return self.UserValves(**raw)
        except Exception:
            return self.UserValves()

    def _text(self, messages: list) -> str:
        parts = []
        for msg in messages:
            c = msg.get("content", "")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
        return " ".join(parts)

    def _has_image(self, messages: list) -> bool:
        return any(
            isinstance(p, dict) and p.get("type") == "image_url"
            for msg in messages
            for p in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        )

    def _has_audio(self, messages: list) -> bool:
        return any(
            isinstance(p, dict) and p.get("type") in ("audio", "audio_url")
            for msg in messages
            for p in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        )

    def _is_code(self, text: str) -> bool:
        return bool(re.search(
            r"\b(code|coding|python|javascript|typescript|java|c\+\+|golang|rust|sql|bash|script|function|class|algorithm|debug|refactor)\b",
            text, re.IGNORECASE,
        )) or "```" in text

    def _resolve(self, model: str) -> str:
        return self.valves.model_default if model in self.valves.unavailable_models else model

    async def _emit(self, emitter, text: str, done: bool = True):
        if emitter and callable(emitter):
            await emitter({"type": "status", "data": {"description": text, "done": done}})

    # ── inlet ──────────────────────────────────────────────────────────────

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        user: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        try:
            messages: list = body.get("messages", [])
            user_info = __user__ or user or {}
            user_id = user_info.get("id", "anonymous")
            uv = self._parse_uv(user_info)

            # ── Если image_gen_pipeline уже обработал запрос — не трогаем ─
            if body.get("__image_handled__") or body.get("__pptx_handled__"):
                return body

            # Latest user message text
            user_msgs = [m for m in messages if m.get("role") == "user"]
            last_text = ""
            if user_msgs:
                lc = user_msgs[-1].get("content", "")
                last_text = lc if isinstance(lc, str) else self._text([user_msgs[-1]])

            full_text = self._text(messages)

            # ── Full memory wipe ───────────────────────────────────────────
            if uv.clear_memory:
                db_clear_all(user_id)
                await self._emit(__event_emitter__, "🗑 Память полностью очищена")

            if uv.enable_memory:
                # ── "Forget X" cluster deletion ────────────────────────────
                topic = _parse_forget_command(last_text)
                if topic:
                    keywords = [w for w in re.split(r"[\s,]+", topic) if len(w) > 2]
                    deleted = db_delete_cluster(user_id, keywords)
                    await self._emit(
                        __event_emitter__,
                        f"🗑 Удалено {deleted} фрагментов памяти о «{topic}»",
                    )
                    # Return early — no need to answer, just confirm
                    body["messages"] = messages
                    return body

                # ── Semantic RAG retrieval ─────────────────────────────────
                relevant = await semantic_search(
                    self.valves.base_url, self.valves.api_key,
                    user_id, last_text or full_text,
                    top_k=self.valves.memory_top_k,
                )

                if relevant:
                    block = _build_knowledge_block(relevant)
                    messages = _inject_knowledge(messages, block)
                    body["messages"] = messages
                    cats = list({c["category"] for c in relevant})
                    await self._emit(
                        __event_emitter__,
                        f"🧠 База знаний: {len(relevant)} фрагментов [{', '.join(cats)}]",
                    )

            # ── Model routing ──────────────────────────────────────────────
            if self._has_audio(messages):
                chosen, label = self._resolve(self.valves.model_audio), "🎙 Аудио"
            elif self._has_image(messages):
                chosen, label = self._resolve(self.valves.model_vision), "👁 Изображение"
            elif self._is_code(full_text):
                chosen, label = self._resolve(self.valves.model_code), "💻 Код"
            else:
                chosen, label = self._resolve(self.valves.model_default), "💬 Текст"

            body["model"] = chosen
            if uv.show_model_status:
                await self._emit(__event_emitter__, f"{label} → {chosen}")

        except Exception as exc:
            log.error("Inlet error: %s", exc)
            await self._emit(__event_emitter__, f"⚠️ Ошибка: {exc}")
            body["model"] = self.valves.model_default

        return body

    # ── outlet ─────────────────────────────────────────────────────────────

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        user: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """
        After each response: split → summarize (mts-anya) → conflict-check → embed → store.
        Runs as a background task so it never delays the response.
        """
        try:
            user_info = __user__ or user or {}
            user_id = user_info.get("id", "anonymous")
            uv = self._parse_uv(user_info)

            if uv.enable_memory:
                messages = body.get("messages", [])
                asyncio.create_task(
                    _outlet_pipeline(
                        self.valves.base_url,
                        self.valves.api_key,
                        self.valves.model_memory,
                        user_id,
                        messages,
                    )
                )
        except Exception as e:
            log.error("Outlet error: %s", e)
        return body


# OpenWebUI Pipelines compatibility alias
Pipeline = Filter
