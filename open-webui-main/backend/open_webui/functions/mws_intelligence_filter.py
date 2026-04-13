"""
MWS Intelligence Filter  v3.0
==============================
Global inlet filter: intent detection + auto model routing + features.

Группы моделей MWS GPT (по каталогу):

  GROUP 1 — Text-only LLM (обычный чат, код, анализ):
    mws-gpt-alpha, deepseek-r1-distill-qwen-32b, gemma-3-27b-it,
    glm-4.6-357b, gpt-oss-120b, gpt-oss-20b, kimi-k2-instruct,
    llama-3.1-8b-instruct, llama-3.3-70b-instruct, qwen2.5-72b-instruct,
    Qwen3-235B-A22B-Instruct-2507-FP8, qwen3-32b, qwen3-coder-480b-a35b,
    QwQ-32B, T-pro-it-1.0

  GROUP 2 — VLM (изображение + текст → текст):
    qwen2.5-vl, qwen2.5-vl-72b, qwen3-vl-30b-a3b-instruct, cotype-pro-vl-32b

  GROUP 3 — Image generation (текст → картинка):
    qwen-image, qwen-image-lightning

  GROUP 4 — ASR / STT (аудио → текст, НЕ для chat completions):
    whisper-medium, whisper-turbo-local

  GROUP 5 — Embeddings (текст → вектор, НЕ для chat completions):
    bge-m3, bge-multilingual-gemma2, qwen3-embedding-8b

  Маршрутизация:
    Изображение в чате  → GROUP 2 (VLM)
    «Нарисуй...»        → GROUP 3 (image gen) или DALL-E pipeline
    Аудиофайл           → STT pipeline → GROUP 1
    Код                 → qwen3-coder-480b-a35b → GROUP 1
    Deep research       → deepseek-r1-distill-qwen-32b / QwQ-32B
    Презентация         → Qwen3-235B / glm-4.6-357b (длинный контекст)
    Поиск/URL           → GROUP 1 + web_search feature
    Обычный текст       → mws-gpt-alpha → лучшая доступная из GROUP 1
"""

import re
import logging
from typing import Any, Optional

from pydantic import BaseModel

log = logging.getLogger(__name__)

# ── GROUP 5: Embeddings — исключить из chat completions ─────────────────────
_EMBEDDING_SUFFIXES = ('embedding', 'embed', 'bge-m3', 'bge-multilingual')
_EMBEDDING_IDS = frozenset({
    'bge-m3', 'bge-multilingual-gemma2', 'BAAI/bge-multilingual-gemma2',
    'qwen3-embedding-8b',
})

# ── GROUP 4: STT — исключить из chat completions ────────────────────────────
_STT_IDS = frozenset({'whisper-medium', 'whisper-turbo-local'})

# ── GROUP 3: Image generation models (используются ТОЛЬКО через /images/generations,
# НЕ через /chat/completions — нельзя ставить как chat-модель) ───────────────
_IMAGE_GEN_MODEL_IDS = [
    'qwen-image-lightning',
    'qwen-image',
]

# ── GROUP 2: VLM priority list ───────────────────────────────────────────────
_VLM_IDS = [
    'qwen2.5-vl-72b',
    'qwen3-vl-30b-a3b-instruct',
    'qwen2.5-vl',
    'cotype-pro-vl-32b',
]

# ── GROUP 1: Text LLM priority lists ────────────────────────────────────────
_CODE_IDS = ['qwen3-coder-480b-a35b']

_REASONING_IDS = ['deepseek-r1-distill-qwen-32b', 'QwQ-32B', 'qwq-32b']

_LONG_CTX_IDS = [
    'Qwen3-235B-A22B-Instruct-2507-FP8',
    'glm-4.6-357b',
    'qwen2.5-72b-instruct',
    'gpt-oss-120b',
    'gpt-oss-20b',
]

# Tool-calling capable models (для web search)
_TOOL_CALL_IDS = [
    'llama-3.3-70b-instruct',
    'qwen2.5-72b-instruct',
    'Qwen3-235B-A22B-Instruct-2507-FP8',
    'glm-4.6-357b',
]

_GENERAL_CHAT_IDS = [
    'mws-gpt-alpha',
    'Qwen3-235B-A22B-Instruct-2507-FP8',
    'qwen2.5-72b-instruct',
    'glm-4.6-357b',
    'gpt-oss-120b',
    'gpt-oss-20b',
    'llama-3.3-70b-instruct',
    'qwen3-32b',
    'kimi-k2-instruct',
    'gemma-3-27b-it',
    'T-pro-it-1.0',
    'llama-3.1-8b-instruct',
    'deepseek-r1-distill-qwen-32b',
    'QwQ-32B',
]

# ── Регулярки для детектирования интентов ───────────────────────────────────
# ПРИНЦИП: только явные команды пользователя, никакой автоматики по контексту.

# Генерация изображения — только если пользователь явно просит нарисовать/сгенерировать
_IMAGE_GEN_RE = re.compile(
    r'(?:^|[\s,\.!?])(?:'
    r'нарисуй|нарисовать|сгенерируй\s+(?:изображение|картинку|рисунок|фото|фотографию)|'
    r'создай\s+(?:изображение|картинку|рисунок|фото|фотографию)|'
    r'сделай\s+(?:картинку|рисунок|изображение|фото|фотографию)|'
    r'generate\s+(?:an?\s+)?(?:image|picture|photo|illustration)|'
    r'create\s+(?:an?\s+)?(?:image|picture|photo|illustration)|'
    r'draw\s+(?:me\s+)?(?:a\s+)?|'
    r'make\s+(?:an?\s+)?(?:image|picture|photo|illustration)|'
    r'paint\s+(?:me\s+)?(?:a\s+)?'
    r')',
    re.IGNORECASE,
)

# Поиск в интернете — только явные команды поиска/нахождения
_WEB_SEARCH_RE = re.compile(
    r'(?:^|[\s,\.!?])(?:'
    r'найди\s+(?:в\s+интернете|онлайн|информацию|ссылки?|сайты?)|'
    r'поищи\s+(?:в\s+интернете|онлайн)?|'
    r'найдите\s+(?:в\s+интернете|онлайн)?|'
    r'поиск\s+(?:в\s+интернете|по\s+интернету)|'
    r'дай\s+ссылк|дайте\s+ссылк|'
    r'найди\s+ссылк|найди\s+сайт|'
    r'где\s+(?:купить|найти|скачать|посмотреть)|'
    r'search\s+(?:the\s+)?(?:web|internet|online)\s+for|'
    r'find\s+(?:online|on\s+the\s+(?:web|internet))|'
    r'look\s+up\s+online|'
    r'give\s+me\s+(?:a\s+)?link|find\s+(?:a\s+)?link|'
    r'where\s+(?:can\s+I\s+)?(?:buy|find|download)'
    r')',
    re.IGNORECASE,
)

# Deep research — только явные команды исследования
_DEEP_RESEARCH_RE = re.compile(
    r'(?:^|[\s,\.!?])(?:'
    r'deep\s+research|in.depth\s+(?:research|analysis|study)|'
    r'comprehensive\s+(?:research|analysis|report)|'
    r'подробно\s+исследуй|глубокое\s+исследование|'
    r'детальный\s+(?:анализ|обзор|отчёт)\s+(?:о|по|на\s+тему)|'
    r'всестороннее\s+исследование|'
    r'research\s*:|исследуй\s*:|глубоко\s+исследуй|'
    r'сделай\s+(?:deep\s+research|глубокий\s+анализ)'
    r')',
    re.IGNORECASE,
)

# Презентация — только явные команды
_PRESENTATION_RE = re.compile(
    r'(?:^|[\s,\.!?])(?:'
    r'сделай\s+презентацию|создай\s+презентацию|подготовь\s+презентацию|'
    r'презентация\s+(?:о|по|на\s+тему)|слайды\s+(?:о|по|на\s+тему)|'
    r'make\s+(?:a\s+)?presentation|create\s+(?:a\s+)?presentation|'
    r'build\s+(?:a\s+)?presentation|generate\s+(?:a\s+)?presentation|'
    r'slides?\s+(?:about|on|for)\b'
    r')',
    re.IGNORECASE,
)

# Код — только явные команды написать/реализовать код
_CODE_RE = re.compile(
    r'(?:^|[\s,\.!?])(?:'
    r'напиши\s+(?:функцию|класс|скрипт|программу|код)|'
    r'написать\s+(?:код|функцию|скрипт|программу)|'
    r'сделай\s+(?:функцию|класс|скрипт|программу)|'
    r'реализуй|отладь|исправь\s+(?:баг|ошибку)|'
    r'write\s+(?:a\s+)?(?:function|class|script|program|code|algorithm)|'
    r'implement\s+(?:a\s+)?(?:function|class|feature)|'
    r'fix\s+(?:the\s+)?(?:bug|error|issue)'
    r')',
    re.IGNORECASE,
)

_URL_RE = re.compile(r'https?://[^\s<>"\']+', re.IGNORECASE)

_MEMORY_FACTS_RE = re.compile(
    r'(?:'
    r'меня зовут|моё имя|мое имя|я\s+\w+\s*,\s*я\s+\w+|'
    r'my name is|i am a\b|i\'m a\b|i work as|'
    r'я работаю|моя профессия|я занимаюсь|'
    r'\bзапомни\b|remember that|не забудь'
    r')',
    re.IGNORECASE,
)

# ── Системные промпты ────────────────────────────────────────────────────────
_PRESENTATION_PROMPT = """You are a professional presentation designer.
Generate a **Marp-compatible Markdown** presentation.
- Front-matter: `---\\nmarp: true\\ntheme: default\\n---`
- Separate slides with `---`
- First slide: title + subtitle; last slide: Summary
- Use `## Heading` for slide titles, bullet points for content (max 5-6 per slide)
- Use **bold** for key terms
Generate 8-12 slides. After slides, offer PPTX export."""

_DEEP_RESEARCH_PROMPT = """You are an expert research analyst. Conduct deep multi-step research.
Steps: search broadly → parse pages → cross-reference → synthesize → cite sources.
Output format:
## Executive Summary
## Key Findings
## Data & Evidence
## Analysis & Insights
## Conclusions
## Sources
[1] URL - Title ...
Always include numbered citations [1], [2], etc. Be thorough and evidence-based."""

_AUDIO_CONTEXT_PROMPT = """The user uploaded an audio file that was automatically transcribed (ASR).
The transcription is in the context. Acknowledge the audio, answer questions about it,
or summarize its content if no specific question was asked."""


# ── Вспомогательные функции ──────────────────────────────────────────────────

def _tail(model_id: str) -> str:
    """Возвращает последний сегмент id (после / и :)."""
    return model_id.split('/')[-1].split(':')[0]


def _is_embedding(model_id: str) -> bool:
    mid_lower = model_id.lower()
    tail_lower = _tail(model_id).lower()
    if model_id in _EMBEDDING_IDS:
        return True
    for s in _EMBEDDING_SUFFIXES:
        if s in mid_lower:
            return True
    return False


def _is_stt(model_id: str) -> bool:
    if model_id in _STT_IDS:
        return True
    tail_lower = _tail(model_id).lower()
    return tail_lower.startswith('whisper')


def _is_usable(model_id: str) -> bool:
    """Модель пригодна для chat completions."""
    return not _is_embedding(model_id) and not _is_stt(model_id)


def _resolve(models: dict, candidates: list) -> Optional[str]:
    """
    Ищет первую доступную модель из списка кандидатов в словаре MODELS.

    Стратегии (в порядке приоритета):
      1. Точное совпадение ключа
      2. Ключ заканчивается на '/<candidate>' или ':<candidate>'
      3. Хвост ключа (после последнего / и :) совпадает с кандидатом
      4. Регистронезависимое совпадение хвоста
    """
    if not models:
        return None

    # Строим вспомогательные индексы один раз
    tail_map: dict[str, list[str]] = {}   # tail_lower → [model_id, ...]
    for mid in models:
        if not _is_usable(mid):
            continue
        t = _tail(mid).lower()
        tail_map.setdefault(t, []).append(mid)

    for c in candidates:
        # 1. Точное совпадение
        if c in models and _is_usable(c):
            return c

    for c in candidates:
        c_lower = c.lower()
        # 2. Суффикс пути
        for mid in models:
            if not _is_usable(mid):
                continue
            if mid.endswith('/' + c) or mid.endswith(':' + c):
                return mid
        # 3 & 4. По хвосту (case-insensitive)
        hits = tail_map.get(c_lower, [])
        if hits:
            return hits[0]

    return None


def _best_chat(models: dict) -> Optional[str]:
    """Лучшая доступная text-only LLM."""
    result = _resolve(models, _GENERAL_CHAT_IDS)
    if result:
        return result
    # Fallback: первая usable модель
    for mid in models:
        if _is_usable(mid) and models[mid].get('owned_by') != 'arena':
            return mid
    return None


def _best_vlm(models: dict) -> Optional[str]:
    """Лучшая доступная VLM."""
    result = _resolve(models, _VLM_IDS)
    if result:
        return result
    # Fallback: любая модель с vision capability
    for mid, m in models.items():
        if not _is_usable(mid):
            continue
        caps = (m.get('info', {}) or {}).get('meta', {}) or {}
        caps = caps.get('capabilities', {}) or {}
        if caps.get('vision', False):
            return mid
    return _best_chat(models)


def _parse_manual_ids(raw: str) -> set:
    if not raw or not raw.strip():
        return set()
    return {x.strip() for x in raw.split(',') if x.strip()}


def _extract_chat_id(metadata: Optional[dict]) -> Optional[str]:
    if not metadata:
        return None
    return metadata.get('chat_id') or metadata.get('session_id')


def _get_per_chat_memory(chat_id: str) -> str:
    """Загружает per-chat память из поля chat.chat['memory']."""
    try:
        from open_webui.models.chats import Chats
        chat = Chats.get_chat_by_id(chat_id)
        if not chat:
            return ''
        entries = (chat.chat or {}).get('memory', [])
        if not entries:
            return ''
        lines = ['[Chat Memory — facts from this conversation:]']
        for e in entries[-20:]:
            lines.append(f'• {e["content"] if isinstance(e, dict) else e}')
        return '\n'.join(lines)
    except Exception as ex:
        log.debug('[MWS] per-chat memory load error: %s', ex)
        return ''


class Filter:
    """
    MWS Intelligence Filter v3.0

    Автоматически:
    - Выбирает модель по типу задачи (VLM / image-gen / code / research / chat)
    - Включает web_search при поисковых запросах или URL
    - Включает deep_research при соответствующих фразах
    - Включает image_generation при запросах «нарисуй...»
    - Инжектирует per-chat память в системный промпт
    - Добавляет контекст для аудиофайлов

    Ручной override: задайте модель в dropdown UI или добавьте её id
    в valve `manual_model_ids` — автоселект будет пропущен.
    """

    class Valves(BaseModel):
        # Автоматика
        auto_model_selection: bool = True
        auto_web_search: bool = True
        auto_image_generation: bool = True
        auto_deep_research: bool = True
        auto_presentation: bool = True
        # Память
        always_inject_memory: bool = True
        inject_per_chat_memory: bool = True
        # Переопределения моделей (пустая строка = авто)
        chat_model: str = ''
        code_model: str = ''
        vision_model: str = ''
        reasoning_model: str = ''
        long_context_model: str = ''
        image_gen_model: str = ''
        # Ручной список id через запятую — не трогать автоселектом
        manual_model_ids: str = ''

    def __init__(self):
        self.valves = self.Valves()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_models(self, request: Any) -> dict:
        """Безопасно получаем словарь MODELS из app.state."""
        if request is None:
            return {}
        models = getattr(request.app.state, 'MODELS', None)
        if not models:
            return {}
        # RedisDict или обычный dict — оба поддерживают .items() / 'in'
        return models

    def _valve_model(self, valve_val: str, models: dict) -> Optional[str]:
        """Резолвим valve-значение (может быть с/без префикса)."""
        if not valve_val:
            return None
        if valve_val in models and _is_usable(valve_val):
            return valve_val
        return _resolve(models, [valve_val])

    def _pick_target(
        self,
        models: dict,
        has_image: bool,
        has_audio: bool,
        is_image_gen: bool,
        is_code: bool,
        is_deep: bool,
        is_presentation: bool,
    ) -> Optional[str]:
        """Выбираем целевую модель по приоритету интентов."""

        # Valve overrides
        if has_image:
            v = self._valve_model(self.valves.vision_model, models)
            return v or _best_vlm(models)

        if is_code:
            v = self._valve_model(self.valves.code_model, models)
            return v or _resolve(models, _CODE_IDS) or _best_chat(models)

        if is_deep:
            v = self._valve_model(self.valves.reasoning_model, models)
            return v or _resolve(models, _REASONING_IDS) or _best_chat(models)

        if is_presentation:
            v = self._valve_model(self.valves.long_context_model, models)
            return v or _resolve(models, _LONG_CTX_IDS) or _best_chat(models)

        if is_image_gen:
            # image-gen модели (qwen-image-lightning, qwen-image) работают ТОЛЬКО через
            # /images/generations — их нельзя ставить как chat-модель.
            # Оставляем текущую chat-модель, feature image_generation включится ниже.
            v = self._valve_model(self.valves.chat_model, models)
            return v or _best_chat(models)

        # Обычный чат
        v = self._valve_model(self.valves.chat_model, models)
        return v or _best_chat(models)

    # ── inlet ─────────────────────────────────────────────────────────────────

    def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __request__: Any = None,
    ) -> dict:
        try:
            messages = body.get('messages', [])
            current_model: str = body.get('model') or ''
            features: dict = body.get('features', {}) or {}

            # ── Парсим последнее сообщение пользователя ───────────────────
            user_text = ''
            has_image = False
            has_audio = False

            for msg in reversed(messages):
                if msg.get('role') != 'user':
                    continue
                content = msg.get('content', '')
                if isinstance(content, str):
                    user_text = content
                elif isinstance(content, list):
                    for part in content:
                        if part.get('type') == 'text':
                            user_text += part.get('text', '')
                        elif part.get('type') == 'image_url':
                            has_image = True
                break

            # ── Проверяем прикреплённые файлы ────────────────────────────
            files = body.get('files', []) or (__metadata__ or {}).get('files', []) or []
            for f in files:
                ct = (f.get('content_type') or f.get('type') or '').lower()
                name = (f.get('name') or f.get('filename') or '').lower()
                if ct.startswith('image/'):
                    has_image = True
                elif ct.startswith('audio/') or any(
                    name.endswith(x) for x in ('.mp3', '.wav', '.ogg', '.m4a', '.flac', '.webm', '.opus')
                ):
                    has_audio = True

            # ── Детектируем интенты — ТОЛЬКО по явным командам пользователя ──
            # URL в тексте НЕ триггерит поиск автоматически
            is_deep = bool(_DEEP_RESEARCH_RE.search(user_text))
            is_presentation = bool(_PRESENTATION_RE.search(user_text))
            is_image_gen = bool(_IMAGE_GEN_RE.search(user_text)) and not has_image
            is_web = bool(_WEB_SEARCH_RE.search(user_text))  # только явный запрос поиска
            is_code = bool(_CODE_RE.search(user_text)) and not has_image

            # Если текущий запрос текстовый (нет изображения в последнем сообщении),
            # зачищаем image_url из истории — text-only модели не умеют их обрабатывать.
            # Это также фиксит ошибку "not a multimodal model" при вопросах после VLM-ответа.
            if not has_image:
                cleaned = []
                stripped = False
                for msg in messages:
                    content = msg.get('content', '')
                    if isinstance(content, list) and any(p.get('type') == 'image_url' for p in content):
                        text_only = ' '.join(p.get('text', '') for p in content if p.get('type') == 'text').strip()
                        msg = dict(msg)
                        msg['content'] = text_only
                        stripped = True
                    cleaned.append(msg)
                if stripped:
                    body['messages'] = cleaned
                    messages = cleaned
                    log.info('[MWS] stripped image_url from history (text-only request)')

            # Если web/deep_research запрос — дополнительно убеждаемся что история чистая
            if is_web or is_deep:
                cleaned = []
                has_images_in_history = False
                for msg in messages:
                    content = msg.get('content', '')
                    if isinstance(content, list):
                        has_images_in_history = any(p.get('type') == 'image_url' for p in content)
                        if has_images_in_history:
                            text_only = ' '.join(p.get('text', '') for p in content if p.get('type') == 'text').strip()
                            msg = dict(msg)
                            msg['content'] = text_only
                    cleaned.append(msg)
                if has_images_in_history:
                    body['messages'] = cleaned
                    messages = cleaned
                    log.info('[MWS] stripped image_url from history for web/research')

            # ── Автовыбор модели ──────────────────────────────────────────
            manual_ids = _parse_manual_ids(self.valves.manual_model_ids)
            skip_auto = current_model and current_model in manual_ids

            if not skip_auto and self.valves.auto_model_selection:
                models = self._get_models(__request__)
                if models:
                    target = self._pick_target(
                        models, has_image, has_audio,
                        is_image_gen, is_code, is_deep, is_presentation,
                    )
                    if target and target in models:
                        old = current_model
                        body['model'] = target
                        log.info(
                            '[MWS] model %s→%s | img=%s vlm=%s code=%s deep=%s deck=%s imggen=%s web=%s url=%s',
                            old, target, has_image, has_image, is_code, is_deep,
                            is_presentation, is_image_gen, is_web, has_url,
                        )
                    elif target is None and is_image_gen:
                        # Нет image-gen модели — оставляем текущую, включим DALL-E ниже
                        pass
                    else:
                        log.debug('[MWS] target=%s not in MODELS (keys sample: %s)',
                                  target, list(models.keys())[:5])

            # ── Включаем фичи ─────────────────────────────────────────────
            if self.valves.auto_deep_research and is_deep:
                features['deep_research'] = True
                features.pop('web_search', None)
                from open_webui.utils.misc import add_or_update_system_message
                body['messages'] = add_or_update_system_message(
                    _DEEP_RESEARCH_PROMPT, body.get('messages', []), append=False
                )
                log.info('[MWS] deep_research enabled')

            elif self.valves.auto_web_search and is_web and not features.get('web_search'):
                features['web_search'] = True
                # При web_search переключаем на text-only модель с tool-calling
                if not skip_auto and self.valves.auto_model_selection:
                    models = self._get_models(__request__)
                    if models:
                        web_model = _resolve(models, _TOOL_CALL_IDS) or _best_chat(models)
                        if web_model and web_model in models:
                            body['model'] = web_model
                            log.info('[MWS] web_search: switched to text model %s', web_model)
                log.info('[MWS] web_search enabled (url=%s)', has_url)

            if self.valves.auto_image_generation and is_image_gen:
                if not features.get('image_generation'):
                    features['image_generation'] = True
                    log.info('[MWS] image_generation enabled')
                # Запрещаем модели писать текст про изображение — картинка уже генерируется
                from open_webui.utils.misc import add_or_update_system_message
                body['messages'] = add_or_update_system_message(
                    'The image is being generated automatically by the image generation system. '
                    'Do NOT describe the image, do NOT say you cannot create images, do NOT add any text. '
                    'Only respond with a very short confirmation like "Generating your image..." or stay silent.',
                    body.get('messages', []), append=False
                )

            if self.valves.auto_presentation and is_presentation:
                features['presentation'] = True
                from open_webui.utils.misc import add_or_update_system_message
                body['messages'] = add_or_update_system_message(
                    _PRESENTATION_PROMPT, body.get('messages', []), append=False
                )
                log.info('[MWS] presentation mode enabled')

            if has_audio and not features.get('audio_context_injected'):
                from open_webui.utils.misc import add_or_update_system_message
                body['messages'] = add_or_update_system_message(
                    _AUDIO_CONTEXT_PROMPT, body.get('messages', []), append=True
                )
                features['audio_context_injected'] = True

            # ── Память ────────────────────────────────────────────────────
            if self.valves.always_inject_memory and not features.get('memory'):
                features['memory'] = True

            if self.valves.inject_per_chat_memory:
                chat_id = _extract_chat_id(__metadata__)
                if chat_id:
                    mem = _get_per_chat_memory(chat_id)
                    if mem:
                        from open_webui.utils.misc import add_or_update_system_message
                        body['messages'] = add_or_update_system_message(
                            mem, body.get('messages', []), append=True
                        )

            body['features'] = features

        except Exception as exc:
            log.exception('[MWS] inlet error: %s', exc)

        return body

    # ── outlet ────────────────────────────────────────────────────────────────

    def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __request__: Any = None,
    ) -> dict:
        """Сохраняем факты о пользователе в per-chat память."""
        try:
            if not self.valves.inject_per_chat_memory:
                return body

            chat_id = _extract_chat_id(__metadata__)
            if not chat_id:
                return body

            # Ищем последнее сообщение пользователя
            last_user = ''
            for msg in reversed(body.get('messages', [])):
                if msg.get('role') != 'user':
                    continue
                c = msg.get('content', '')
                if isinstance(c, str):
                    last_user = c
                elif isinstance(c, list):
                    for p in c:
                        if p.get('type') == 'text':
                            last_user += p.get('text', '')
                break

            if not last_user or len(last_user) > 500:
                return body
            if not _MEMORY_FACTS_RE.search(last_user):
                return body

            try:
                import time as _t
                from open_webui.models.chats import Chats
                chat_model = Chats.get_chat_by_id(chat_id)
                if not chat_model:
                    return body
                chat_data = dict(chat_model.chat or {})
                entries = list(chat_data.get('memory', []))
                existing = [e.get('content', e) if isinstance(e, dict) else e for e in entries]
                if last_user not in existing:
                    entries.append({'content': last_user[:300], 'ts': int(_t.time())})
                    chat_data['memory'] = entries[-50:]
                    Chats.update_chat_by_id(chat_id, chat_data)
                    log.debug('[MWS] saved per-chat memory fact, chat_id=%s', chat_id)
            except Exception as ex:
                log.debug('[MWS] per-chat memory save error: %s', ex)

        except Exception as exc:
            log.exception('[MWS] outlet error: %s', exc)

        return body
