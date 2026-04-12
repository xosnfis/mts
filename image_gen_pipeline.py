"""
Image Generation Pipeline — OpenWebUI Filter.

Перехватывает запросы на генерацию изображений.
Генерирует ОДНУ картинку через MWS API, отправляет через event_emitter.
LLM получает заглушку и возвращает пустой ответ — никакого текста от модели.
"""

import re
import json
import logging
import httpx
from typing import Optional, Callable, Awaitable
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

_IMAGE_RE = re.compile(
    # 1. Прямые глаголы рисования/визуализации — все формы лица и времени
    r"\b(нарисуй|нарисуйте|нарисуем|нарисую|нарисует|нарисовать|нарисуй-ка"
    r"|изобрази|изобразите|изобразим|изображу|изобразить"
    r"|визуализируй|визуализируйте|визуализируем|визуализировать"
    r"|покажи|покажите|покажем|покажу|покажет"
    r"|нарисуй-ка|набросай|набросок|зарисуй"
    r"|draw|paint|illustrate|sketch|render|depict|visualize|show\s+me|generate\s+image)\b"
    r"|"
    # 2. Глаголы генерации/создания + визуальный объект (с любым текстом между ними до 60 символов)
    r"\b(сгенерируй|сгенерируем|сгенерирую|сгенерирует|сгенерировать|сгенерируйте"
    r"|создай|создадим|создам|создаст|создать|создайте"
    r"|сделай|сделаем|сделаю|сделает|сделать|сделайте"
    r"|придумай|придумаем|придумаю|придумать|придумайте"
    r"|скинь|скинуть|пришли|прислать"
    r"|нужно\s+(?:сгенерировать|создать|нарисовать|сделать|придумать)"
    r"|можешь\s+(?:сгенерировать|нарисовать|создать|сделать|нарисовать)"
    r"|хочу\s+(?:картинку|изображение|фото|логотип|рисунок|арт|иллюстрацию)"
    r"|generate|create|make|produce|design|build|get\s+me)\s+.{0,60}"
    r"(картинк|изображени|фотографи|фото|рисунок|арт\b|иллюстраци|логотип|лого\b|баннер|обложк|аватар|иконк|постер|плакат|обои|превью|миниатюр"
    r"|image|picture|photo|illustration|logo|banner|cover|avatar|icon|poster|artwork|graphic|wallpaper|thumbnail|preview)"
    r"|"
    # 3. Прямой запрос с "для [файл]" — "нарисую для img2.jpg чтобы там..."
    r"\b(нарисую|нарисуй|сгенерируй|создай|сделай|покажи)\s+(?:мне\s+)?(?:для\s+)?[\w\-]+\.(?:jpg|jpeg|png|webp|gif|svg)\b",
    re.IGNORECASE | re.DOTALL,
)

# Паттерны для редактирования/дополнения предыдущей картинки
_EDIT_RE = re.compile(
    r"\b(добавь|добавить|добавим|добавлю"
    r"|дорисуй|дорисовать|дорисуем|дорисую"
    r"|измени|изменить|изменим|изменю"
    r"|поставь|вставь|допиши|вставить"
    r"|сделай\s+(?:его|её|их|фон|цвет|фоном|ярче|темнее|больше|меньше)"
    r"|убери|удали|замени|перекрась|перекрасить"
    r"|теперь\s+(?:добавь|сделай|нарисуй|убери|измени)"
    r"|ещё\s+(?:добавь|нарисуй|сделай)"
    r"|add|include|put|insert|place|also\s+add|now\s+add|remove|change|replace|make\s+it|update\s+it)\b",
    re.IGNORECASE,
)


def _is_image_request(text: str) -> bool:
    return bool(_IMAGE_RE.search(text))


# Признаки того что запрос касается кода — для составных задач
_CODE_CONTEXT_RE = re.compile(
    r"\b(код|code|html|css|javascript|js|скрипт|script|файл|file|функци|function"
    r"|лендинг|landing|сайт|site|страниц|page|компонент|component|стил|style"
    r"|который\s+(?:ты\s+)?(?:писал|написал|делал|сделал|генерировал|создавал)"
    r"|в\s+(?:код|html|файл|скрипт|стил)"
    r"|в\s+этот\s+(?:код|файл|html|скрипт)"
    r"|выше|ранее|предыдущ)\b",
    re.IGNORECASE,
)

# Союзы/конструкции указывающие на составной запрос
_COMPOUND_RE = re.compile(
    r"\b(и\s+(?:также\s+)?(?:добавь|вставь|обнови|измени|дополни|включи|замени)"
    r"|а\s+(?:также\s+)?(?:добавь|вставь|обнови|измени)"
    r"|после\s+(?:чего|этого)\s+(?:добавь|вставь|обнови|измени|дополни)"
    r"|затем\s+(?:добавь|вставь|обнови|измени)"
    r"|потом\s+(?:добавь|вставь|обнови|измени)"
    r"|and\s+(?:also\s+)?(?:add|insert|update|change|put)\s+it"
    r"|then\s+(?:add|insert|update|put)\s+it)\b",
    re.IGNORECASE,
)


def _is_code_edit_request(text: str) -> bool:
    """Запрос на изменение кода — есть edit-глагол И контекст про код."""
    return bool(_EDIT_RE.search(text)) and bool(_CODE_CONTEXT_RE.search(text))


def _is_compound_request(text: str) -> bool:
    """Составной запрос: сгенерируй картинку И добавь её в код."""
    return (
        _is_image_request(text)
        and bool(_CODE_CONTEXT_RE.search(text))
        and bool(_COMPOUND_RE.search(text) or _EDIT_RE.search(text))
    )


def _is_edit_request(text: str) -> bool:
    """Запрос на дополнение картинки — edit-глагол, не image-запрос, не code-запрос."""
    return (
        bool(_EDIT_RE.search(text))
        and not _is_image_request(text)
        and not _is_code_edit_request(text)
    )


def _last_image_prompt(messages: list) -> Optional[str]:
    """Fallback: ищет промпт в истории сообщений если self._last_prompt недоступен."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and "imagegen.gpt.mws.ru" in content:
                m = re.search(r"!\[([^\]]{2,200})\]", content)
                if m:
                    return m.group(1)
    return None


def _clean_prompt(text: str) -> str:
    """Strip command words, keep the actual subject."""
    s = re.sub(
        r"^[\s]*(?:пожалуйста[,\s]*)?"
        r"(?:нарисуй|нарисуйте|изобрази|визуализируй|сгенерируй|создай|сделай"
        r"|draw|paint|illustrate|sketch|render|generate|create|make|depict)"
        r"[\s,]*(?:мне[,\s]*)?(?:пожалуйста[,\s]*)?",
        "", text.strip(), flags=re.IGNORECASE,
    )
    s = re.sub(
        r"^(?:картинку?|изображение|фото|рисунок|арт|иллюстрацию?"
        r"|image|picture|photo|illustration|artwork)\s+(?:of\s+|про\s+|с\s+)?",
        "", s.strip(), flags=re.IGNORECASE,
    )
    return s.strip() or text.strip()


class Filter:
    type = "filter"
    class Valves(BaseModel):
        model_config = {"protected_namespaces": ()}
        pipelines: list[str] = Field(default=["*"])
        priority: int = Field(default=1)
        base_url: str = "https://api.gpt.mws.ru/v1"
        api_key: str = "sk-ewgiaPC3A6pPDYHwR8siVA"
        image_model: str = "qwen-image-lightning"
        image_size: str = "1024x1024"
        translate_model: str = "llama-3.3-70b-instruct"
        translate_prompt: bool = True

    class UserValves(BaseModel):
        enable_image_gen: bool = Field(
            default=True,
            description="🎨 Генерировать изображения по запросу",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._pending: dict[str, dict] = {}    # user_id → {url, raw_prompt, error}
        self._last_prompt: dict[str, str] = {} # user_id → последний успешный промпт

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

    def _msg_text(self, msg: dict) -> str:
        c = msg.get("content", "")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
        return ""

    async def _emit(self, emitter, text: str, done: bool = True):
        if emitter and callable(emitter):
            await emitter({"type": "status", "data": {"description": text, "done": done}})

    async def _translate(self, text: str) -> str:
        if not re.search(r"[а-яёА-ЯЁ]", text):
            return text
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(
                    f"{self.valves.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.valves.api_key}"},
                    json={
                        "model": self.valves.translate_model,
                        "messages": [{"role": "user", "content":
                            f"Translate to English for image generation. Return ONLY the prompt:\n{text}"}],
                        "temperature": 0,
                        "max_tokens": 150,
                    },
                )
            r.raise_for_status()
            result = r.json()["choices"][0]["message"]["content"].strip()
            log.info("Translated: '%s' → '%s'", text, result)
            return result
        except Exception as e:
            log.error("Translation failed: %s", e)
            return text

    async def _generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(
                f"{self.valves.base_url}/images/generations",
                headers={"Authorization": f"Bearer {self.valves.api_key}"},
                json={
                    "model": self.valves.image_model,
                    "prompt": prompt,
                    "n": 1,
                    "size": self.valves.image_size,
                },
            )
        r.raise_for_status()
        url = r.json()["data"][0]["url"]
        log.info("Image generated: %s", url)
        return url

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        user: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        user_info = __user__ or user or {}
        uv = self._parse_uv(user_info)
        user_id = user_info.get("id", "anonymous")

        if not uv.enable_image_gen:
            return body

        messages: list = body.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return body

        last_text = self._msg_text(user_msgs[-1])

        # Определяем тип запроса
        is_gen = _is_image_request(last_text)
        # Edit: есть слово "добавь/add/..." И у пользователя есть предыдущий промпт
        prev_prompt = self._last_prompt.get(user_id) or _last_image_prompt(messages)
        is_edit = not is_gen and _is_edit_request(last_text) and prev_prompt is not None

        if not is_gen and not is_edit:
            return body

        # ── Запрос на картинку — перехватываем ────────────────────────────
        body["__image_handled__"] = True
        # Сохраняем оригинальные сообщения до подмены заглушкой
        body["_original_messages"] = [m.copy() for m in messages]
        await self._emit(__event_emitter__, "🎨 Генерирую...", done=False)

        try:
            if is_edit:
                # Объединяем предыдущий промпт с новым дополнением
                addition = _clean_prompt(last_text)
                raw_prompt = f"{prev_prompt}, {addition}"
                log.info("Edit: base='%s' + add='%s'", prev_prompt, addition)
            else:
                raw_prompt = _clean_prompt(last_text)

            if self.valves.translate_prompt:
                prompt = await self._translate(raw_prompt)
            else:
                prompt = raw_prompt

            url = await self._generate(prompt)

            # Сохраняем в памяти экземпляра — outlet заберёт
            self._pending[user_id] = {
                "url": url,
                "raw_prompt": raw_prompt,
                "compound": _is_compound_request(last_text),
                "original_messages": body.get("_original_messages", messages),
                "original_text": last_text,
            }
            # Запоминаем промпт для будущих edit-запросов
            self._last_prompt[user_id] = raw_prompt
            await self._emit(__event_emitter__, "✅ Готово")

        except Exception as exc:
            log.error("Image gen error: %s", exc)
            self._pending[user_id] = {"error": str(exc)}
            await self._emit(__event_emitter__, f"❌ Ошибка: {exc}")

        # Заглушка для LLM — модель вернёт минимальный ответ
        body["messages"] = [
            {"role": "system", "content": "Reply with a single dot and nothing else."},
            {"role": "user", "content": "."},
        ]
        body["model"] = self.valves.translate_model
        body["stream"] = False

        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        user: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """Подменяем ответ LLM на картинку. Для составных запросов — второй шаг: вставка в код."""
        user_info = __user__ or user or {}
        user_id = user_info.get("id", "anonymous")

        pending = self._pending.pop(user_id, None)
        if not pending:
            return body

        messages = body.get("messages", [])

        if "url" in pending:
            url = pending["url"]
            raw_prompt = pending.get("raw_prompt", "")
            md = f"![{raw_prompt}]({url})\n\n[🔍 Открыть]({url})"

            # Вставляем картинку как ответ ассистента
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    msg["content"] = md
                    break
            else:
                messages.append({"role": "assistant", "content": md})
            body["messages"] = messages

            # ── Составной запрос: второй шаг — вставить URL в код ─────────
            if pending.get("compound"):
                await self._emit(__event_emitter__, "✏️ Вставляю в код...", done=False)
                original_msgs = pending.get("original_messages", [])
                original_text = pending.get("original_text", "")
                try:
                    # Строим историю: оригинальные сообщения + картинка как ответ + инструкция
                    code_messages = original_msgs + [
                        {"role": "assistant", "content": md},
                        {
                            "role": "user",
                            "content": (
                                f"Теперь возьми код из нашего разговора выше и замени "
                                f"соответствующий src/placeholder на этот URL: {url}\n"
                                f"Верни полный обновлённый HTML-код."
                            ),
                        },
                    ]
                    async with httpx.AsyncClient(timeout=60) as client:
                        resp = await client.post(
                            f"{self.valves.base_url}/chat/completions",
                            headers={"Authorization": f"Bearer {self.valves.api_key}"},
                            json={
                                "model": self.valves.translate_model,
                                "messages": code_messages,
                                "temperature": 0.1,
                                "max_tokens": 3000,
                            },
                        )
                    resp.raise_for_status()
                    code_reply = resp.json()["choices"][0]["message"]["content"].strip()
                    # Добавляем код как второй блок в ответ
                    for msg in reversed(messages):
                        if msg.get("role") == "assistant":
                            msg["content"] = md + "\n\n---\n\n" + code_reply
                            break
                    body["messages"] = messages
                    await self._emit(__event_emitter__, "✅ Код обновлён")
                except Exception as e:
                    log.error("Compound code-update error: %s", e)
                    await self._emit(__event_emitter__, f"⚠️ Не удалось обновить код: {e}")

        elif "error" in pending:
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    msg["content"] = f"❌ Ошибка генерации: {pending['error']}"
                    break

        return body


# OpenWebUI Pipelines compatibility alias
Pipeline = Filter
