# MWS GPTHub

## Запуск

```bash
docker compose up --build -d
```

Открыть в браузере: [http://localhost:3000](http://localhost:3000)

## Остановка

```bash
docker compose down
```

## Обновление пайплайнов

После изменения `.py` файлов пересобери образ:

```bash
docker compose up --build -d pipelines
```
