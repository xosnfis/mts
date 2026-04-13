# Open WebUI — MWS GPT Edition

Полнофункциональный AI-чат на базе [Open WebUI](https://github.com/open-webui/open-webui), настроенный для работы с [MWS GPT API](https://api.gpt.mws.ru). Никакого Ollama, никаких локальных моделей — всё через облако.

## Быстрый старт

```bash
MWS_GPT_API_KEY=sk-ewgiaPC3A6pPDYHwR8siVA docker compose -f docker-compose.mws.yaml up -d
```

Откройте [http://localhost:3000](http://localhost:3000)

> Порт можно переопределить: `OPEN_WEBUI_PORT=8080 MWS_GPT_API_KEY=sk-... docker compose ...`

## Настройка через .env (опционально)

```bash
cp .env.example .env
# Заполните MWS_GPT_API_KEY и при необходимости WEBUI_SECRET_KEY
docker compose -f docker-compose.mws.yaml up -d
```

## Возможности

| # | Фича |
|---|------|
| 1 | Текстовый чат — стриминг, история |
| 2 | Голосовой чат — STT (whisper-medium) + TTS (tts-1) |
| 3 | Генерация изображений — qwen-image-lightning / DALL-E |
| 4 | Загрузка аудио + автоматический ASR |
| 5 | Vision-модели — qwen2.5-vl-72b и др. |
| 6 | RAG / Файлы — PDF, DOCX, TXT, CSV (bge-m3 embeddings) |
| 7 | Поиск в интернете — DuckDuckGo, без API-ключа |
| 8 | Веб-парсинг по ссылке — автодетект URL |
| 9 | Долгосрочная память — per-user + per-chat |
| 10 | Автовыбор модели — MWS Intelligence Filter |
| 11 | Ручной выбор модели — dropdown в UI |
| 12 | Markdown + подсветка кода |
| 13 | Deep Research — многошаговый режим с источниками |

## Переменные окружения

| Переменная | Обязательна | Описание |
|---|---|---|
| `MWS_GPT_API_KEY` | ✅ | API-ключ MWS GPT |
| `WEBUI_SECRET_KEY` | рекомендуется | Секрет для подписи сессий |
| `WEBUI_AUTH` | нет (default: `true`) | Отключить авторизацию: `false` |
| `OPEN_WEBUI_PORT` | нет (default: `3000`) | Порт на хосте |

## Стек

- Frontend: SvelteKit + Tailwind CSS
- Backend: Python 3.11, FastAPI, SQLAlchemy, Alembic
- База данных: SQLite (данные в Docker volume `open-webui-mws-data`)
- API: [MWS GPT](https://api.gpt.mws.ru) (OpenAI-совместимый)

## Разработка

```bash
# Backend
bash backend/dev.sh

# Frontend (в отдельном терминале)
npm install
npm run dev
```
