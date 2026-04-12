"""
title: Deep Research
id: deep_research_pipeline
description: Режим глубокого исследования. Используй /research <тема> или фразы "глубокий анализ", "подробный отчёт". Выполняет итеративный поиск через deepseek-r1-distill-qwen-32b и возвращает структурированный отчёт.
author: GPTHub
version: 1.0.0
"""

import re
import json
import logging
import asyncio
from typing import Optional, Callable, Awaitable
from urllib.parse import quote_plus

import httpx
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------

_DEEP_RE = re.compile(
    r"^/research\b"
    r"|"
    r"\b(глубокий\s+анализ|глубокое\s+исследование|подробный\s+отчёт|подробный\s+доклад"
    r"|детальный\s+анализ|детальное\s+исследование|comprehensive\s+report"
    r"|deep\s+research|in[\-\s]depth\s+(?:analysis|research|report)"
    r"|исследуй\s+подробно|изучи\s+подробно|напиши\s+(?:подробный\s+)?(?:доклад|реферат|отчёт))\b",
    re.IGNORECASE,
)


def _is_deep_research(text: str) -> bool:
    return bool(_DEEP_RE.search(text))


def _strip_trigger(text: str) -> str:
    """Remove /research prefix and trigger phrases, return clean topic."""
    text = re.sub(r"^/research\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(
        r"\b(глубокий\s+анализ|глубокое\s+исследование|подробный\s+отчёт|подробный\s+доклад"
        r"|детальный\s+анализ|детальное\s+исследование|comprehensive\s+report"
        r"|deep\s+research|in[\-\s]depth\s+(?:analysis|research|report)"
        r"|исследуй\s+подробно|изучи\s+подробно|напиши\s+(?:подробный\s+)?(?:доклад|реферат|отчёт))\b",
        "", text, flags=re.IGNORECASE,
    )
    return re.sub(r"\s{2,}", " ", text).strip(" ,.")


# ---------------------------------------------------------------------------
# Web search (DDG HTML — same approach as web_search_pipeline)
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript|head)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
import html as _html_mod


def _clean_html(raw: str, max_chars: int = 3000) -> str:
    text = _SCRIPT_STYLE_RE.sub(" ", raw)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _html_mod.unescape(text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text[:max_chars]


async def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    try:
        encoded = quote_plus(query)
        async with httpx.AsyncClient(
            timeout=15, follow_redirects=True,
            headers={"User-Agent": ua, "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8"},
        ) as client:
            r = await client.get(f"https://html.duckduckgo.com/html/?q={encoded}&kl=ru-ru")
        r.raise_for_status()
        raw = r.text
        link_re = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
        snippet_re = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
        titles_urls = link_re.findall(raw)
        snippets = [_clean_html(s, 400) for s in snippet_re.findall(raw)]
        results = []
        from urllib.parse import unquote
        for i, (href, title_html) in enumerate(titles_urls[:max_results]):
            real_url = href
            uddg = re.search(r"uddg=([^&]+)", href)
            if uddg:
                real_url = unquote(uddg.group(1))
            title = _clean_html(title_html, 120)
            snippet = snippets[i] if i < len(snippets) else ""
            if real_url and title:
                results.append({"title": title, "url": real_url, "snippet": snippet})
        return results
    except Exception as e:
        log.warning("DDG search error: %s", e)
        return []


async def _searxng_search(base_url: str, query: str, max_results: int = 5) -> list[dict]:
    try:
        encoded = quote_plus(query)
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(
                f"{base_url.rstrip('/')}/search?q={encoded}&format=json&language=auto",
                headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearch/1.0)"},
            )
        r.raise_for_status()
        data = r.json()
        return [
            {"title": i.get("title", ""), "url": i.get("url", ""), "snippet": i.get("content", "")}
            for i in data.get("results", [])[:max_results]
        ]
    except Exception as e:
        log.error("SearXNG error: %s", e)
        return []


async def _search(query: str, searxng_url: str = "", max_results: int = 5) -> list[dict]:
    if searxng_url:
        results = await _searxng_search(searxng_url, query, max_results)
        if results:
            return results
    return await _ddg_search(query, max_results)


def _format_results(results: list[dict]) -> str:
    if not results:
        return "Результаты не найдены."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        if r.get("url"):
            lines.append(f"   URL: {r['url']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet'][:300]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM calls (deepseek-r1-distill-qwen-32b)
# ---------------------------------------------------------------------------

async def _llm(
    base_url: str, api_key: str, model: str,
    messages: list, max_tokens: int = 2000, temperature: float = 0.3,
) -> str:
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error("LLM call error: %s", e)
        return ""


async def _generate_subquestions(
    base_url: str, api_key: str, model: str, topic: str, n: int = 4
) -> list[str]:
    """Ask the reasoning model to decompose the topic into N search sub-questions."""
    prompt = (
        f"Тема для глубокого исследования: «{topic}»\n\n"
        f"Сгенерируй ровно {n} конкретных поисковых вопроса, которые вместе дадут "
        f"полное понимание темы. Каждый вопрос — отдельная строка, без нумерации и маркеров. "
        f"Только вопросы, без пояснений."
    )
    raw = await _llm(
        base_url, api_key, model,
        [{"role": "user", "content": prompt}],
        max_tokens=400, temperature=0.4,
    )
    questions = [q.strip(" -•*") for q in raw.splitlines() if q.strip()]
    # Filter out <think> blocks that deepseek-r1 may emit
    questions = [q for q in questions if not q.startswith("<") and len(q) > 10]
    return questions[:n]


async def _synthesize(
    base_url: str, api_key: str, model: str,
    topic: str, evidence_blocks: list[str], iteration: int,
) -> str:
    """Synthesize gathered evidence into intermediate findings."""
    evidence = "\n\n---\n\n".join(evidence_blocks)
    prompt = (
        f"Ты проводишь итерацию {iteration} глубокого исследования темы: «{topic}».\n\n"
        f"Вот собранные данные из веб-поиска:\n\n{evidence}\n\n"
        f"Синтезируй ключевые факты, выяви пробелы в знаниях, "
        f"сформулируй 2-3 уточняющих вопроса для следующей итерации. "
        f"Отвечай на русском языке."
    )
    return await _llm(
        base_url, api_key, model,
        [{"role": "user", "content": prompt}],
        max_tokens=1500, temperature=0.3,
    )


async def _write_report(
    base_url: str, api_key: str, model: str,
    topic: str, all_evidence: str,
) -> str:
    """Write the final comprehensive Markdown report."""
    prompt = (
        f"На основе собранных данных напиши подробный структурированный отчёт по теме: «{topic}».\n\n"
        f"Данные:\n{all_evidence}\n\n"
        f"Требования к отчёту:\n"
        f"- Заголовок H1 с темой\n"
        f"- Введение (2-3 абзаца)\n"
        f"- Основные разделы с подзаголовками H2/H3\n"
        f"- Конкретные факты, цифры, примеры\n"
        f"- Раздел «Выводы»\n"
        f"- Раздел «Источники» со ссылками в формате [название](url)\n"
        f"- Язык: русский\n"
        f"- Объём: не менее 800 слов\n\n"
        f"Пиши только отчёт, без вводных фраз типа «Вот отчёт:»."
    )
    return await _llm(
        base_url, api_key, model,
        [{"role": "user", "content": prompt}],
        max_tokens=4000, temperature=0.2,
    )


# ---------------------------------------------------------------------------
# Main deep research orchestrator
# ---------------------------------------------------------------------------

async def run_deep_research(
    base_url: str,
    api_key: str,
    model: str,
    topic: str,
    iterations: int = 2,
    subquestions_per_iter: int = 4,
    searxng_url: str = "",
    emitter: Optional[Callable] = None,
) -> str:
    """
    Orchestrates the full deep research loop.
    Returns a Markdown report string.
    """

    async def emit(text: str, done: bool = False):
        if emitter and callable(emitter):
            await emitter({"type": "status", "data": {"description": text, "done": done}})

    all_evidence: list[str] = []
    all_urls: list[str] = []

    for iteration in range(1, iterations + 1):
        await emit(f"🔬 Итерация {iteration}/{iterations}: генерирую вопросы…")

        # Step 1: generate sub-questions
        context_so_far = "\n\n".join(all_evidence[-3:]) if all_evidence else ""
        topic_with_context = topic if not context_so_far else (
            f"{topic}\n\nУже известно:\n{context_so_far[:800]}"
        )
        questions = await _generate_subquestions(
            base_url, api_key, model, topic_with_context, subquestions_per_iter
        )
        if not questions:
            questions = [topic]

        # Step 2: search for each sub-question
        iter_evidence: list[str] = []
        for qi, question in enumerate(questions, 1):
            await emit(f"🔍 [{iteration}/{iterations}] Поиск {qi}/{len(questions)}: «{question[:60]}»…")
            results = await _search(question, searxng_url, max_results=5)
            if results:
                block = f"Вопрос: {question}\n\n{_format_results(results)}"
                iter_evidence.append(block)
                all_evidence.append(block)
                for r in results:
                    if r.get("url") and r["url"] not in all_urls:
                        all_urls.append(r["url"])
            await asyncio.sleep(0.5)  # polite delay

        # Step 3: synthesize iteration findings
        if iter_evidence and iteration < iterations:
            await emit(f"🧠 [{iteration}/{iterations}] Синтезирую находки…")
            synthesis = await _synthesize(
                base_url, api_key, model, topic, iter_evidence, iteration
            )
            if synthesis:
                all_evidence.append(f"[Синтез итерации {iteration}]\n{synthesis}")

    # Final report
    await emit("📝 Пишу итоговый отчёт…")
    combined_evidence = "\n\n---\n\n".join(all_evidence)
    report = await _write_report(base_url, api_key, model, topic, combined_evidence)

    if not report:
        report = f"# {topic}\n\nНе удалось сгенерировать отчёт. Попробуйте позже."

    await emit("✅ Отчёт готов", done=True)
    return report


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class Pipeline:
    type = "filter"
    class Valves(BaseModel):
        model_config = {"protected_namespaces": ()}
        pipelines: list[str] = Field(default=["*"])
        priority: int = Field(default=0, description="Filter priority — lower runs first")

        base_url: str = Field(default="https://api.gpt.mws.ru/v1", description="LLM API base URL")
        api_key: str = Field(default="sk-ewgiaPC3A6pPDYHwR8siVA", description="API key")
        research_model: str = Field(
            default="deepseek-r1-distill-qwen-32b",
            description="Модель для глубокого исследования (reasoning)",
        )
        searxng_url: str = Field(default="", description="SearXNG URL (опционально)")
        iterations: int = Field(default=2, description="Количество итераций поиска (1-4)")
        subquestions: int = Field(default=4, description="Подвопросов на итерацию (2-6)")

    class UserValves(BaseModel):
        enable_deep_research: bool = Field(
            default=True,
            description="🔬 Режим глубокого исследования (/research или ключевые слова)",
        )

    def __init__(self):
        self.valves = self.Valves()

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

    def _last_user_text(self, messages: list) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                c = msg.get("content", "")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return " ".join(
                        p.get("text", "") for p in c
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
        return ""

    async def _emit(self, emitter, text: str, done: bool = False):
        if emitter and callable(emitter):
            await emitter({"type": "status", "data": {"description": text, "done": done}})

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        user: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        try:
            user_info = __user__ or user or {}
            uv = self._parse_uv(user_info)

            if not uv.enable_deep_research:
                return body

            messages: list = body.get("messages", [])
            last_text = self._last_user_text(messages)

            if not last_text or not _is_deep_research(last_text):
                return body

            topic = _strip_trigger(last_text)
            if not topic:
                topic = last_text

            await self._emit(__event_emitter__, f"🔬 Deep Research: «{topic[:60]}»…")

            report = await run_deep_research(
                base_url=self.valves.base_url,
                api_key=self.valves.api_key,
                model=self.valves.research_model,
                topic=topic,
                iterations=max(1, min(4, self.valves.iterations)),
                subquestions_per_iter=max(2, min(6, self.valves.subquestions)),
                searxng_url=self.valves.searxng_url,
                emitter=__event_emitter__,
            )

            # Inject report as system context so the LLM just echoes/formats it
            report_block = (
                "[DEEP RESEARCH REPORT]\n"
                "Ниже — готовый отчёт. Выведи его пользователю ДОСЛОВНО без изменений.\n\n"
                + report
            )
            # Replace messages with a minimal pass-through
            body["messages"] = [
                {"role": "system", "content": report_block},
                {"role": "user", "content": "Выведи отчёт."},
            ]
            body["model"] = self.valves.research_model
            body["__deep_research_done__"] = True

        except Exception as exc:
            log.error("Deep research error: %s", exc)
            await self._emit(__event_emitter__, f"❌ Ошибка: {exc}", done=True)

        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        user: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        return body
