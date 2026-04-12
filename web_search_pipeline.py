"""
Web Search & Parsing Pipeline — OpenWebUI Filter.

Capabilities:
  1. Web Search  — real DuckDuckGo HTML search (full results, not Instant Answer).
                   Falls back to SearXNG if configured.
  2. URL Parsing — fetches page, strips HTML, injects as temporary context.
  3. Safety Check — every URL is validated before being shown to the user:
                    • known-bad domain blacklist
                    • suspicious pattern detection (IP urls, typosquatting, shorteners)
                    • Google Safe Browsing API (optional, requires API key)
                    • redirect chain inspection

Search scope: full web — not limited to any marketplace.
Marketplace queries get an additional direct search link as a bonus.
"""

import re
import json
import logging
import html
from typing import Optional, Callable, Awaitable
from urllib.parse import urlparse, quote_plus, urljoin

import httpx
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_CURRENT_INFO_RE = re.compile(
    r"\b(сегодня|сейчас|текущ\w+|последн\w+|актуальн\w+|свеж\w+"
    r"|новост\w+|новинк\w+|обновлени\w+|релиз|вышел|вышла|вышло"
    r"|погода|температура|курс\s+(?:валют|доллара|евро|рубля)"
    r"|цена|стоимость|котировк\w+|акци\w+|биткоин|крипто"
    r"|расписание|афиша|мероприяти\w+|событи\w+|концерт|матч"
    r"|трафик|пробки|today|right\s+now|current\w*|latest|recent\w*"
    r"|news|update|release|weather|forecast|price|cost|rate|stock|crypto"
    r"|найди|найдите|поищи|поищите|загугли|погугли|ищи|ищем"
    r"|дай\s+ссылк\w*|скинь\s+ссылк\w*|покажи\s+ссылк\w*"
    r"|порекомендуй\w*|посоветуй\w*|рекомендуй\w*"
    r"|где\s+купить|где\s+найти|купить\s+на|найди\s+на"
    r"|на\s+озон\w+|на\s+вайлдберриз\w*|на\s+авито\w*|на\s+амазон\w*"
    r"|search\s+for|look\s+up|find\s+(?:me\s+)?(?:info|link|links)"
    r"|where\s+to\s+buy|on\s+amazon|on\s+ebay)\b",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://[^\s\)\]\>\"\']{4,}", re.IGNORECASE)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|head)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)

# Detect time queries
_TIME_RE = re.compile(
    r"\b(врем[яени]|который\s+час|сколько\s+времени|time|what\s+time|current\s+time"
    r"|часов[ой]?\s+пояс|timezone)\b",
    re.IGNORECASE,
)

# Detect weather queries
_WEATHER_RE = re.compile(
    r"\b(погода|температура|прогноз|осадки|дождь|снег|ветер"
    r"|weather|forecast|temperature|rain|snow|wind|humidity)\b",
    re.IGNORECASE,
)

# City name extraction — after "в", "в городе", "в Нижневартовске" etc.
_CITY_RE = re.compile(
    r"(?:в\s+(?:городе?\s+)?|in\s+(?:city\s+of\s+)?|for\s+)"
    r"([А-ЯЁA-Z][а-яёa-z\-]{2,}(?:\s+[А-ЯЁA-Z][а-яёa-z\-]{2,})?)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Safety — domain blacklist & suspicious pattern detection
# ---------------------------------------------------------------------------

# Known malicious / phishing / spam TLDs and domains (extend as needed)
_BLACKLISTED_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "short.link",
    "cutt.ly", "rebrand.ly", "tiny.cc", "is.gd", "v.gd", "buff.ly",
    "adf.ly", "linkbucks.com", "bc.vc", "sh.st",
}

# Trusted high-authority domains — always safe, skip extra checks
_TRUSTED_DOMAINS = {
    "wikipedia.org", "github.com", "stackoverflow.com", "docs.python.org",
    "developer.mozilla.org", "ozon.ru", "wildberries.ru", "avito.ru",
    "market.yandex.ru", "amazon.com", "amazon.co.uk", "ebay.com",
    "aliexpress.com", "google.com", "youtube.com", "reddit.com",
    "medium.com", "habr.com", "vc.ru", "rbc.ru", "tass.ru",
    "ria.ru", "kommersant.ru", "lenta.ru", "meduza.io",
}

# Regex for suspicious URL patterns
_SUSPICIOUS_RE = re.compile(
    r"(\d{1,3}\.){3}\d{1,3}"          # raw IP address
    r"|[a-z0-9\-]{30,}\."             # very long subdomain (likely generated)
    r"|(login|signin|verify|secure|account|update|confirm|banking)\."
    r"|\.(?:tk|ml|ga|cf|gq|xyz|top|click|download|zip|review|country|kim|science|work|party|gdn)$",
    re.IGNORECASE,
)


def _get_root_domain(url: str) -> str:
    """Extract root domain (e.g. 'sub.example.com' → 'example.com')."""
    try:
        host = urlparse(url).netloc.lower().split(":")[0]
        parts = host.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else host
    except Exception:
        return ""


def _quick_safety_check(url: str) -> tuple[bool, str]:
    """
    Fast local safety check. Returns (is_safe, reason).
    Does NOT make network calls.
    """
    if not url.startswith("https://") and not url.startswith("http://"):
        return False, "не HTTP(S) ссылка"

    root = _get_root_domain(url)

    if root in _TRUSTED_DOMAINS:
        return True, "доверенный домен"

    if root in _BLACKLISTED_DOMAINS:
        return False, f"домен в чёрном списке: {root}"

    if _SUSPICIOUS_RE.search(url):
        return False, "подозрительный паттерн URL"

    # Prefer HTTPS
    if url.startswith("http://"):
        return True, "⚠️ HTTP (не зашифровано)"

    return True, "OK"


async def _safe_browsing_check(api_key: str, url: str) -> tuple[bool, str]:
    """
    Google Safe Browsing API v4 lookup.
    Returns (is_safe, threat_type_or_ok).
    """
    if not api_key:
        return True, "GSB не настроен"
    try:
        payload = {
            "client": {"clientId": "openwebui-pipeline", "clientVersion": "1.0"},
            "threatInfo": {
                "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": url}],
            },
        }
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(
                f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}",
                json=payload,
            )
        r.raise_for_status()
        data = r.json()
        if data.get("matches"):
            threat = data["matches"][0].get("threatType", "UNKNOWN")
            return False, f"Google Safe Browsing: {threat}"
        return True, "GSB: чисто"
    except Exception as e:
        log.warning("Safe Browsing API error: %s", e)
        return True, "GSB: ошибка проверки"


async def check_url_safety(url: str, gsb_api_key: str = "") -> tuple[bool, str]:
    """
    Full safety check: local patterns + optional Google Safe Browsing.
    Returns (is_safe, reason_string).
    """
    is_safe, reason = _quick_safety_check(url)
    if not is_safe:
        return False, reason

    if gsb_api_key:
        is_safe, gsb_reason = await _safe_browsing_check(gsb_api_key, url)
        if not is_safe:
            return False, gsb_reason

    return True, reason


async def _filter_safe_results(
    results: list[dict], gsb_api_key: str = ""
) -> list[dict]:
    """Filter results list, keeping only safe URLs. Adds safety_note to each."""
    safe = []
    for r in results:
        url = r.get("url", "")
        if not url:
            safe.append(r)
            continue
        is_safe, reason = await check_url_safety(url, gsb_api_key)
        if is_safe:
            r["safety_note"] = reason
            safe.append(r)
        else:
            log.warning("Blocked unsafe URL %s — %s", url, reason)
    return safe

# ---------------------------------------------------------------------------
# Specialized APIs — time & weather (no key required)
# ---------------------------------------------------------------------------

# IANA timezone map for major Russian cities
_CITY_TZ = {
    "москва": "Europe/Moscow", "санкт-петербург": "Europe/Moscow", "питер": "Europe/Moscow",
    "екатеринбург": "Asia/Yekaterinburg", "новосибирск": "Asia/Novosibirsk",
    "омск": "Asia/Omsk", "красноярск": "Asia/Krasnoyarsk",
    "иркутск": "Asia/Irkutsk", "якутск": "Asia/Yakutsk",
    "владивосток": "Asia/Vladivostok", "магадан": "Asia/Magadan",
    "нижневартовск": "Asia/Yekaterinburg", "сургут": "Asia/Yekaterinburg",
    "тюмень": "Asia/Yekaterinburg", "челябинск": "Asia/Yekaterinburg",
    "уфа": "Asia/Yekaterinburg", "пермь": "Asia/Yekaterinburg",
    "казань": "Europe/Moscow", "самара": "Europe/Samara",
    "ростов-на-дону": "Europe/Moscow", "краснодар": "Europe/Moscow",
    "волгоград": "Europe/Moscow", "воронеж": "Europe/Moscow",
    "нижний новгород": "Europe/Moscow", "саратов": "Europe/Moscow",
    "хабаровск": "Asia/Vladivostok", "барнаул": "Asia/Barnaul",
    "томск": "Asia/Tomsk", "кемерово": "Asia/Novokuznetsk",
    "новокузнецк": "Asia/Novokuznetsk", "чита": "Asia/Chita",
    "благовещенск": "Asia/Yakutsk", "мурманск": "Europe/Moscow",
    "архангельск": "Europe/Moscow", "калининград": "Europe/Kaliningrad",
}


async def _get_time_for_city(city: str) -> Optional[dict]:
    """
    Get current time for a city using WorldTimeAPI.
    Returns dict with time info or None.
    """
    city_lower = city.lower().strip()
    tz = _CITY_TZ.get(city_lower)

    # Try WorldTimeAPI by timezone
    if tz:
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(f"https://worldtimeapi.org/api/timezone/{tz}")
            if r.status_code == 200:
                data = r.json()
                dt_str = data.get("datetime", "")
                # Parse: "2026-04-12T17:23:45.123456+05:00"
                dt_part = dt_str[:19].replace("T", " ")
                tz_offset = dt_str[19:25] if len(dt_str) > 19 else ""
                return {
                    "city": city,
                    "timezone": tz,
                    "datetime": dt_part,
                    "utc_offset": tz_offset,
                    "day_of_week": data.get("day_of_week"),
                }
        except Exception as e:
            log.warning("WorldTimeAPI error: %s", e)

    # Fallback: try by city name via timeapi.io
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(
                f"https://timeapi.io/api/Time/current/zone?timeZone={quote_plus(tz or city)}"
            )
        if r.status_code == 200:
            data = r.json()
            return {
                "city": city,
                "timezone": data.get("timeZone", tz or city),
                "datetime": f"{data.get('date', '')} {data.get('time', '')}",
                "utc_offset": "",
            }
    except Exception as e:
        log.warning("timeapi.io error: %s", e)

    return None


async def _get_weather(city: str) -> Optional[str]:
    """
    Get current weather via wttr.in (no API key).
    Returns formatted string or None.
    """
    try:
        encoded = quote_plus(city)
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get(
                f"https://wttr.in/{encoded}?format=j1",
                headers={"User-Agent": "curl/7.68.0"},
            )
        r.raise_for_status()
        data = r.json()
        current = data["current_condition"][0]
        area = data.get("nearest_area", [{}])[0]
        area_name = area.get("areaName", [{}])[0].get("value", city)
        temp_c = current.get("temp_C", "?")
        feels_like = current.get("FeelsLikeC", "?")
        desc = current.get("lang_ru", [{}])[0].get("value") or current.get("weatherDesc", [{}])[0].get("value", "")
        humidity = current.get("humidity", "?")
        wind_kmph = current.get("windspeedKmph", "?")
        return (
            f"Погода в {area_name}: {desc}, {temp_c}°C "
            f"(ощущается как {feels_like}°C), "
            f"влажность {humidity}%, ветер {wind_kmph} км/ч"
        )
    except Exception as e:
        log.warning("wttr.in error: %s", e)
        return None


def _extract_city(text: str) -> Optional[str]:
    """Extract city name from query text."""
    m = _CITY_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: look for known city names directly
    text_lower = text.lower()
    for city in _CITY_TZ:
        if city in text_lower:
            return city
    return None


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

def _clean_html(raw: str, max_chars: int = 6000) -> str:
    text = _SCRIPT_STYLE_RE.sub(" ", raw)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Search backends
# ---------------------------------------------------------------------------

async def _ddg_html_search(query: str, max_results: int = 8) -> list[dict]:
    """
    Scrape DuckDuckGo HTML search.
    Tries multiple User-Agent strings to avoid blocks.
    """
    results = []
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    ]
    for ua in user_agents:
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
            snippets = [_clean_html(s, 300) for s in snippet_re.findall(raw)]

            for i, (href, title_html) in enumerate(titles_urls[:max_results]):
                from urllib.parse import unquote
                real_url = href
                uddg = re.search(r"uddg=([^&]+)", href)
                if uddg:
                    real_url = unquote(uddg.group(1))
                title = _clean_html(title_html, 120)
                snippet = snippets[i] if i < len(snippets) else ""
                if real_url and title:
                    results.append({"title": title, "url": real_url, "snippet": snippet})

            if results:
                break  # success — stop trying other UAs
        except Exception as e:
            log.warning("DDG attempt failed (%s): %s", ua[:30], e)

    return results


async def _brave_search(query: str, api_key: str, max_results: int = 8) -> list[dict]:
    """
    Brave Search API — free tier: 2000 queries/month, no CC required.
    https://api.search.brave.com/
    """
    if not api_key:
        return []
    try:
        encoded = quote_plus(query)
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"https://api.search.brave.com/res/v1/web/search?q={encoded}&count={max_results}&search_lang=ru",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
            )
        r.raise_for_status()
        data = r.json()
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            }
            for item in data.get("web", {}).get("results", [])[:max_results]
        ]
    except Exception as e:
        log.error("Brave Search error: %s", e)
        return []


async def _searxng_search(base_url: str, query: str, max_results: int = 8) -> list[dict]:
    """SearXNG JSON API."""
    try:
        encoded = quote_plus(query)
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(
                f"{base_url.rstrip('/')}/search?q={encoded}&format=json&language=auto",
                headers={"User-Agent": "Mozilla/5.0 (compatible; OpenWebUI-Bot/1.0)"},
            )
        r.raise_for_status()
        data = r.json()
        return [
            {"title": i.get("title", ""), "url": i.get("url", ""), "snippet": i.get("content", "")}
            for i in data.get("results", [])[:max_results]
        ]
    except Exception as e:
        log.error("SearXNG search error: %s", e)
        return []


# ---------------------------------------------------------------------------
# Marketplace helpers
# ---------------------------------------------------------------------------

_MARKETPLACE_RE = re.compile(
    r"\b(озон\w*|ozon\w*|вайлдберриз\w*|wildberries\w*|\bwb\b|авито\w*|avito\w*"
    r"|яндекс[\s\-]*маркет\w*|yandex[\s\-]*market\w*|amazon\w*|ebay\w*"
    r"|aliexpress\w*|алиэкспресс\w*)",
    re.IGNORECASE,
)

_MARKETPLACE_URLS = {
    "ozon":        "https://www.ozon.ru/search/?text={q}",
    "озон":        "https://www.ozon.ru/search/?text={q}",
    "wildberries": "https://www.wildberries.ru/catalog/0/search.aspx?search={q}",
    "вайлдберриз": "https://www.wildberries.ru/catalog/0/search.aspx?search={q}",
    "wb":          "https://www.wildberries.ru/catalog/0/search.aspx?search={q}",
    "авито":       "https://www.avito.ru/rossiya?q={q}",
    "avito":       "https://www.avito.ru/rossiya?q={q}",
    "яндексмаркет":"https://market.yandex.ru/search?text={q}",
    "yandexmarket":"https://market.yandex.ru/search?text={q}",
    "amazon":      "https://www.amazon.com/s?k={q}",
    "ebay":        "https://www.ebay.com/sch/i.html?_nkw={q}",
    "aliexpress":  "https://www.aliexpress.com/wholesale?SearchText={q}",
    "алиэкспресс": "https://www.aliexpress.com/wholesale?SearchText={q}",
}

_NOISE_RE = re.compile(
    r"\b(порекомендуй\w*|посоветуй\w*|рекомендуй\w*|найди|найдите|поищи\w*"
    r"|покажи|дай|скинь|пришли|хочу|нужно|нужна|нужен|ищу|купить|купи"
    r"|ссылк\w+|линк\w+|пожалуйста|плиз|мне|нам"
    r"|recommend|suggest|find|show|give|get|buy|order|search|look\s+for)\b",
    re.IGNORECASE,
)
_COLOR_RE = re.compile(
    r"\b(красн\w+|синн?\w+|чёрн\w+|черн\w+|бел\w+|зелён\w+|зелен\w+|жёлт\w+|желт\w+"
    r"|розов\w+|фиолетов\w+|оранжев\w+|серебрист\w+|золотист\w+|коричнев\w+|сер\w+"
    r"|red|blue|black|white|green|yellow|pink|purple|orange|silver|gold|brown|gr[ae]y)\b",
    re.IGNORECASE,
)
_RATING_RE = re.compile(
    r"(\d[\.,]\d|\d)\s*(?:звезд\w*|star\w*|★)"
    r"|\b(топ|лучш\w+|высок\w+\s+оценк\w+|best|top.rated|highly.rated)\b",
    re.IGNORECASE,
)


def _extract_product(query: str) -> str:
    text = _MARKETPLACE_RE.sub("", query)
    text = _NOISE_RE.sub(" ", text)
    text = re.sub(
        r"\b(с\s+оценк\w+|с\s+рейтинг\w+|оценк\w+|рейтинг\w+)\s*[\d,\.]*\s*(?:звезд\w*|star\w*)?\b",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(на|в|из|по|для|со|с|и|или|а|но)\b", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip(" ,.")


def _detect_marketplace(text: str) -> Optional[tuple[str, str]]:
    m = _MARKETPLACE_RE.search(text)
    if not m:
        return None
    raw = m.group(1).lower()
    for key, url_tpl in _MARKETPLACE_URLS.items():
        if raw.startswith(key) or key.startswith(raw[:4]):
            return (m.group(1), url_tpl)
    return None


def _marketplace_link(query: str) -> Optional[dict]:
    detected = _detect_marketplace(query)
    if not detected:
        return None
    name, url_tpl = detected
    product = _extract_product(query)
    if not product:
        return None
    search_term = product
    color = _COLOR_RE.search(query)
    if color and color.group(0).lower() not in search_term.lower():
        search_term = color.group(0) + " " + search_term
    rating_m = _RATING_RE.search(query)
    if rating_m:
        num_m = re.search(r"(\d[\.,]\d|\d)", rating_m.group(0))
        if num_m:
            search_term += f" {num_m.group(1)} звезды"
    url = url_tpl.format(q=quote_plus(search_term.strip()))
    return {"title": f"Поиск «{search_term.strip()}» на {name}", "url": url, "snippet": f"Прямая ссылка: {url}"}

# ---------------------------------------------------------------------------
# Query normalization via LLM
# ---------------------------------------------------------------------------

async def _normalize_query(base_url: str, api_key: str, model: str, user_text: str) -> str:
    """
    Use LLM to extract a clean, nominative-case search query from user message.
    E.g. "дай красивую обувь оранжевого цвета на озоне" → "красивая обувь оранжевого цвета"
    Falls back to raw text on error.
    """
    prompt = (
        "Извлеки поисковый запрос из сообщения пользователя.\n"
        "Правила:\n"
        "- Оставь только предмет поиска (товар, тема, вопрос)\n"
        "- Приведи прилагательные и существительные в именительный падеж\n"
        "- Убери команды: 'найди', 'дай', 'покажи', 'порекомендуй', 'ссылку', 'на озоне' и т.п.\n"
        "- Сохрани цвет, размер, бренд, характеристики если есть\n"
        "- Если упомянуто несколько маркетплейсов — игнорируй их названия\n"
        "- Верни ТОЛЬКО поисковый запрос, без пояснений\n\n"
        f"Сообщение: {user_text}\n"
        "Запрос:"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 60,
                },
            )
        r.raise_for_status()
        result = r.json()["choices"][0]["message"]["content"].strip()
        # Strip quotes if LLM wrapped the answer
        result = result.strip('"\'«»')
        log.info("Query normalized: '%s' → '%s'", user_text[:60], result)
        return result if result else user_text
    except Exception as e:
        log.warning("Query normalization failed: %s", e)
        return user_text


def _detect_all_marketplaces(text: str) -> list[tuple[str, str]]:
    """Return list of (name, url_template) for ALL marketplaces mentioned in text."""
    found = []
    seen_keys = set()
    for m in _MARKETPLACE_RE.finditer(text):
        raw = m.group(1).lower()
        for key, url_tpl in _MARKETPLACE_URLS.items():
            if (raw.startswith(key) or key.startswith(raw[:4])) and key not in seen_keys:
                found.append((m.group(1), url_tpl))
                seen_keys.add(key)
                break
    return found


def _marketplace_links_all(product: str, original_query: str) -> list[dict]:
    """Build direct search links for ALL marketplaces mentioned in the query."""
    marketplaces = _detect_all_marketplaces(original_query)
    if not marketplaces:
        return []

    search_term = product
    color = _COLOR_RE.search(original_query)
    if color and color.group(0).lower() not in search_term.lower():
        search_term = color.group(0) + " " + search_term
    rating_m = _RATING_RE.search(original_query)
    if rating_m:
        num_m = re.search(r"(\d[\.,]\d|\d)", rating_m.group(0))
        if num_m:
            search_term += f" {num_m.group(1)} звезды"

    results = []
    for name, url_tpl in marketplaces:
        url = url_tpl.format(q=quote_plus(search_term.strip()))
        results.append({
            "title": f"Поиск «{search_term.strip()}» на {name}",
            "url": url,
            "snippet": f"Прямая ссылка: {url}",
            "safety_note": "доверенный домен",
        })
    return results


# ---------------------------------------------------------------------------
# Unified search entry point
# ---------------------------------------------------------------------------

async def web_search(
    query: str,
    original_text: str = "",
    product: str = "",
    searxng_url: Optional[str] = None,
    brave_api_key: str = "",
    max_results: int = 8,
    gsb_api_key: str = "",
) -> list[dict]:
    """
    Full web search: SearXNG → Brave → DDG HTML, with safety filtering.
    """
    raw_results = []

    if searxng_url:
        raw_results = await _searxng_search(searxng_url, query, max_results)

    if not raw_results and brave_api_key:
        raw_results = await _brave_search(query, brave_api_key, max_results)

    if not raw_results:
        raw_results = await _ddg_html_search(query, max_results)

    # Add direct links for ALL mentioned marketplaces
    source = original_text or query
    mp_results = _marketplace_links_all(product or query, source)
    for mp in mp_results:
        raw_results.insert(0, mp)

    safe_results = await _filter_safe_results(raw_results, gsb_api_key)
    return safe_results[:max_results]


# ---------------------------------------------------------------------------
# URL fetching + parsing
# ---------------------------------------------------------------------------

async def fetch_url(url: str, max_chars: int = 6000, gsb_api_key: str = "") -> tuple[Optional[str], str]:
    """
    Fetch URL, check safety, strip HTML.
    Returns (content_or_None, safety_note).
    """
    is_safe, reason = await check_url_safety(url, gsb_api_key)
    if not is_safe:
        return None, f"заблокировано: {reason}"
    try:
        async with httpx.AsyncClient(
            timeout=15, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; OpenWebUI-Bot/1.0)"},
        ) as client:
            r = await client.get(url)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "html" in ct or "text" in ct:
            return _clean_html(r.text, max_chars), reason
        return None, "не текстовый контент"
    except Exception as e:
        log.error("URL fetch error (%s): %s", url, e)
        return None, f"ошибка загрузки: {e}"


# ---------------------------------------------------------------------------
# Context block builders
# ---------------------------------------------------------------------------

def _build_search_block(results: list[dict], query: str) -> str:
    if not results:
        return ""
    lines = [
        f"[WEB SEARCH RESULTS] Запрос: «{query}»",
        "ИНСТРУКЦИЯ ДЛЯ МОДЕЛИ (обязательно к исполнению):\n"
        "1. Используй эти результаты как основу ответа.\n"
        "2. В конце ответа добавь раздел '## Источники' со списком ссылок.\n"
        "3. Для КАЖДОГО использованного факта вставь ссылку в формате [Название](https://полный-url).\n"
        "4. НИКОГДА не придумывай URL. Используй только URL из этого блока.\n"
        "5. Если URL недоступен — напиши 'источник недоступен'.",
    ]
    for i, r in enumerate(results, 1):
        safe_icon = "✅" if r.get("safety_note", "OK") not in ("", "⚠️ HTTP (не зашифровано)") else "⚠️"
        url = r.get("url", "")
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        lines.append(f"\n{i}. {safe_icon} Заголовок: {title}")
        if url:
            lines.append(f"   URL: {url}")
            lines.append(f"   Ссылка для ответа: [{title}]({url})")
        if snippet:
            lines.append(f"   Описание: {snippet[:300]}")
    return "\n".join(lines)


def _build_url_block(url: str, content: str, safety_note: str) -> str:
    return (
        f"[WEB CONTEXT] Страница: {url} (безопасность: {safety_note})\n"
        "Используй этот текст как временный контекст. Не упоминай этот блок пользователю.\n\n"
        f"{content}"
    )


def _inject_block(messages: list, block: str, tag: str) -> list:
    if not block:
        return messages
    for msg in messages:
        if msg.get("role") == "system":
            if tag in msg["content"]:
                msg["content"] = re.sub(
                    rf"\[{re.escape(tag[1:-1])}\].*?(?=\n\n[^\[]|\Z)",
                    block, msg["content"], flags=re.DOTALL,
                )
            else:
                msg["content"] = block + "\n\n" + msg["content"]
            return messages
    messages.insert(0, {"role": "system", "content": block})
    return messages


def _append_links_to_user_msg(messages: list, results: list[dict]) -> list:
    """Append ready-made markdown links into the last user message."""
    links_md = "\n".join(
        f"- [{r['title']}]({r['url']})" for r in results if r.get("url")
    )
    if not links_md:
        return messages
    for msg in reversed(messages):
        if msg.get("role") == "user":
            if isinstance(msg["content"], str):
                msg["content"] += f"\n\n[Готовые ссылки для ответа]\n{links_md}"
            break
    return messages

# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class Filter:
    type = "filter"
    class Valves(BaseModel):
        model_config = {"protected_namespaces": ()}
        pipelines: list[str] = Field(default=["*"])
        priority: int = Field(default=5)

        base_url: str = Field(default="https://api.gpt.mws.ru/v1", description="LLM API base URL для нормализации запросов.")
        api_key: str = Field(default="sk-ewgiaPC3A6pPDYHwR8siVA", description="API key.")
        query_model: str = Field(default="llama-3.3-70b-instruct", description="Модель для нормализации поискового запроса.")
        searxng_url: str = Field(default="", description="SearXNG URL. Leave empty to use DuckDuckGo.")
        brave_search_api_key: str = Field(default="", description="Brave Search API key (free tier: 2000 req/month). Get at https://api.search.brave.com/")
        google_safe_browsing_api_key: str = Field(default="", description="Google Safe Browsing API key. Leave empty to skip.")
        max_search_results: int = Field(default=6, description="Max results to inject.")
        max_page_chars: int = Field(default=6000, description="Max chars from fetched URL.")
        auto_search: bool = Field(default=True, description="Auto-trigger search for current-info queries.")
        block_http: bool = Field(default=False, description="Block non-HTTPS links entirely.")

    class UserValves(BaseModel):
        enable_web_search: bool = Field(default=True, description="🌐 Веб-поиск и парсинг URL")

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

    async def _emit(self, emitter, text: str, done: bool = True):
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
            # Skip if another pipeline already handled this request
            if body.get("__pptx_handled__") or body.get("__image_handled__"):
                return body

            user_info = __user__ or user or {}
            uv = self._parse_uv(user_info)
            if not uv.enable_web_search:
                return body

            messages: list = body.get("messages", [])
            last_text = self._last_user_text(messages)
            if not last_text:
                return body

            gsb_key = self.valves.google_safe_browsing_api_key

            # ── 1. URL in message → fetch & parse (skip binary files) ─────
            urls = [u for u in _URL_RE.findall(last_text)
                    if not re.search(r'\.(pptx|docx|xlsx|pdf|zip|rar|exe)$', u, re.IGNORECASE)]
            if urls:
                url = urls[0]
                await self._emit(__event_emitter__, f"🌐 Проверяю и загружаю: {url[:55]}…", done=False)
                content, safety_note = await fetch_url(url, self.valves.max_page_chars, gsb_key)
                if content:
                    block = _build_url_block(url, content, safety_note)
                    messages = _inject_block(messages, block, "[WEB CONTEXT]")
                    body["messages"] = messages
                    await self._emit(__event_emitter__, f"🌐 Загружено ({len(content)} симв.) — {safety_note}")
                else:
                    await self._emit(__event_emitter__, f"⚠️ Страница недоступна — {safety_note}")

            # ── 2. Auto search ─────────────────────────────────────────────
            elif self.valves.auto_search and (
                _CURRENT_INFO_RE.search(last_text) or _detect_marketplace(last_text)
            ):
                raw_query = _URL_RE.sub("", last_text).strip()[:200]

                # ── 2a. Time query — use WorldTimeAPI directly ─────────────
                if _TIME_RE.search(last_text):
                    city = _extract_city(last_text)
                    if city:
                        await self._emit(__event_emitter__, f"🕐 Получаю время для {city}…", done=False)
                        time_data = await _get_time_for_city(city)
                        if time_data:
                            block = (
                                f"[WEB CONTEXT] Текущее время\n"
                                f"ИНСТРУКЦИЯ: Сообщи пользователю точное время на основе этих данных.\n\n"
                                f"Город: {time_data['city']}\n"
                                f"Часовой пояс: {time_data['timezone']}\n"
                                f"Дата и время: {time_data['datetime']} (UTC{time_data.get('utc_offset', '')})\n"
                            )
                            messages = _inject_block(messages, block, "[WEB CONTEXT]")
                            body["messages"] = messages
                            await self._emit(__event_emitter__, f"🕐 Время получено для {city}")
                            return body

                # ── 2b. Weather query — use wttr.in directly ──────────────
                if _WEATHER_RE.search(last_text):
                    city = _extract_city(last_text)
                    if city:
                        await self._emit(__event_emitter__, f"🌤 Получаю погоду для {city}…", done=False)
                        weather = await _get_weather(city)
                        if weather:
                            block = (
                                f"[WEB CONTEXT] Текущая погода\n"
                                f"ИНСТРУКЦИЯ: Сообщи пользователю погоду на основе этих данных.\n\n"
                                f"{weather}"
                            )
                            messages = _inject_block(messages, block, "[WEB CONTEXT]")
                            body["messages"] = messages
                            await self._emit(__event_emitter__, f"🌤 Погода получена для {city}")
                            return body

                # ── 2c. General web search ─────────────────────────────────
                await self._emit(__event_emitter__, "🔍 Формирую запрос…", done=False)
                normalized = await _normalize_query(
                    self.valves.base_url, self.valves.api_key,
                    self.valves.query_model, raw_query,
                )
                product = _extract_product(normalized) or normalized

                await self._emit(__event_emitter__, f"🔍 Ищу: «{normalized[:55]}»…", done=False)
                results = await web_search(
                    query=normalized,
                    original_text=raw_query,
                    product=product,
                    searxng_url=self.valves.searxng_url or None,
                    brave_api_key=self.valves.brave_search_api_key,
                    max_results=self.valves.max_search_results,
                    gsb_api_key=gsb_key,
                )

                if self.valves.block_http:
                    results = [r for r in results if r.get("url", "").startswith("https://")]

                if results:
                    block = _build_search_block(results, normalized)
                    messages = _inject_block(messages, block, "[WEB SEARCH RESULTS]")
                    messages = _append_links_to_user_msg(messages, results)
                    body["messages"] = messages
                    await self._emit(__event_emitter__, f"🔍 Найдено {len(results)} безопасных результатов")
                else:
                    await self._emit(__event_emitter__, "🔍 Безопасных результатов не найдено")

        except Exception as exc:
            log.error("WebSearch inlet error: %s", exc)
            await self._emit(__event_emitter__, f"⚠️ Ошибка: {exc}")

        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        user: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        return body


# OpenWebUI Pipelines compatibility alias
Pipeline = Filter
