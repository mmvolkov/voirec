# Voirec

Утилита для транскрибации аудио с поддержкой трёх моделей и опциональной диаризацией (разделением по спикерам). Работает через CLI или HTTP API.

## Модели

| Транскрибер | Модель по умолчанию | Язык | Движок |
|---|---|---|---|
| `whisper` | `onnx-community/whisper-large-v3-turbo` | 99 языков | ONNX (HuggingFace) |
| `gigaam` | `gigaam-v3-e2e-rnnt` | Русский | ONNX |
| `parakeet` | `nemo-parakeet-tdt-0.6b-v3` | Многоязычный | ONNX (NVIDIA NeMo) |

Доступные варианты:

- **whisper**: любая ONNX-модель с HuggingFace (например, `onnx-community/whisper-tiny`, `onnx-community/whisper-large-v3`)
- **gigaam**: `gigaam-v3-ctc`, `gigaam-v3-rnnt`, `gigaam-v3-e2e-ctc`, `gigaam-v3-e2e-rnnt` — `e2e`-варианты включают пунктуацию и нормализацию
- **parakeet**: `nemo-parakeet-tdt-0.6b-v3`, `nemo-parakeet-tdt-0.6b-v2`, `nemo-parakeet-ctc-0.6b-v3`, `nemo-parakeet-rnnt-1.1b-v3`

Все модели скачиваются автоматически при первом использовании и кэшируются в `$HF_HOME` (по умолчанию `~/.cache/huggingface`).

## Установка

```bash
nix-shell
uv sync
```

Для диаризации на моно-аудио нужны дополнительные зависимости:

```bash
uv sync --extra diarize
```

## CLI

### Базовое использование

```bash
# Все три транскрибера
voirec audio.mp3

# Только один
voirec --skip-gigaam --skip-parakeet audio.mp3

# С диаризацией (разделение по спикерам)
voirec --diarize audio.mp3
voirec --diarize --num-speakers 2 audio.mp3
```

### Опции

| Опция | Описание | По умолчанию |
|---|---|---|
| `-o, --output-dir PATH` | Директория для результатов | Рядом с файлом |
| `--whisper-model TEXT` | Модель Whisper (любая HuggingFace ONNX) | `onnx-community/whisper-large-v3-turbo` |
| `--gigaam-model TEXT` | Вариант GigaAM | `gigaam-v3-e2e-rnnt` |
| `--parakeet-model TEXT` | Вариант Parakeet | `nemo-parakeet-tdt-0.6b-v3` |
| `--skip-whisper` | Пропустить Whisper | — |
| `--skip-gigaam` | Пропустить GigaAM | — |
| `--skip-parakeet` | Пропустить Parakeet | — |
| `--language TEXT` | Код языка (`ru`, `en` и т.д.) | автодетект |
| `--diarize` | Включить диаризацию | — |
| `--num-speakers N` | Точное число спикеров | auto |
| `--max-speakers N` | Максимальное число спикеров | — |

### Выходные файлы

```
audio_whisper.txt
audio_gigaam.txt
audio_parakeet.txt

# При --diarize:
audio_dialogue.txt   # с тайм-кодами и метками спикеров
```

Формат диалога:
```
[00:00:01] СПИКЕР_1: Привет, как дела?
[00:00:04] СПИКЕР_2: Всё хорошо, спасибо.
```

## HTTP API

### Запуск

```bash
# Без аутентификации
voirec-api

# С аутентификацией
VOIREC_API_KEYS=token1,token2 voirec-api --port 8080
```

Флаги: `--host` (default: `0.0.0.0`), `--port` (default: `8000`), `--reload`.

Swagger UI доступен по адресу `http://localhost:8000/docs`.

### Аутентификация

Управляется переменными окружения:

- `VOIREC_API_KEYS` — токены через запятую: `token1,token2`
- `VOIREC_API_KEYS_FILE` — путь к файлу с токенами (по одному на строку), приоритет над `VOIREC_API_KEYS`

Если ни одна переменная не задана — аутентификация отключена.

Токен передаётся в заголовке: `Authorization: Bearer <token>` или `X-API-Key: <token>`.

Эндпоинт `/health` всегда доступен без аутентификации.

### Эндпоинты

**`GET /health`**
```json
{"status": "ok", "version": "0.0.1"}
```

**`GET /models`**
Список доступных транскрибберов и вариантов моделей.

**`POST /transcribe`** — multipart form

| Поле | Тип | Обязательный | Описание |
|---|---|---|---|
| `file` | file | Да | Аудиофайл |
| `transcriber` | str | Нет | `whisper` / `gigaam` / `parakeet` (default: `whisper`) |
| `model` | str | Нет | Имя модели (default: дефолт транскриббера) |
| `language` | str | Нет | Код языка, например `ru`, `en` (default: автодетект) |
| `textonly` | bool | Нет | Вернуть plain text вместо JSON (default: `false`) |
| `diarize` | bool | Нет | Диаризация (default: `false`) |
| `num_speakers` | int | Нет | Точное число спикеров |
| `max_speakers` | int | Нет | Максимальное число спикеров |

```bash
curl -F "file=@audio.mp3" -F "transcriber=whisper" http://localhost:8000/transcribe
```

Ответ:
```json
{
  "transcriber": "whisper",
  "model": "onnx-community/whisper-large-v3-turbo",
  "language": null,
  "text": "Привет, мир...",
  "diarized": false
}
```

С `textonly=true` возвращается только текст без JSON-обёртки:
```bash
curl -F "file=@audio.mp3" -F "textonly=true" http://localhost:8000/transcribe
```
```
Привет, мир...
```

## Контейнер

```bash
# CPU (по умолчанию)
podman-compose up -d --build

# NVIDIA GPU
podman-compose --profile cuda up -d --build

# AMD GPU (ROCm)
podman-compose --profile rocm up -d --build

# С аутентификацией через .env
cp .env.example .env   # заполни VOIREC_API_KEYS
podman-compose up -d --build
```

Образ собирается в два этапа (`python:3.12-slim-bookworm` builder → runtime) — оба на Debian для совместимости с manylinux-колёсами onnxruntime. Модели кэшируются в named volume `hf_cache` и не скачиваются повторно при перезапуске.

Для AMD GPU может потребоваться добавить пользователя в группы: `sudo usermod -aG video,render $USER`. При проблемах с определением карты (RDNA3 и старые ROCm) раскомментировать `HSA_OVERRIDE_GFX_VERSION` в `podman-compose.yml`.

## Структура проекта

```
voirec/
├── src/voirec/
│   ├── __init__.py
│   ├── cli.py           # CLI (Click)
│   ├── transcribers.py  # Транскрибберы + диаризация
│   └── api.py           # HTTP API (FastAPI)
├── Containerfile        # Multi-stage Debian образ (CPU)
├── Containerfile.cuda   # То же, с onnxruntime-gpu
├── Containerfile.rocm   # То же, с onnxruntime-rocm
├── podman-compose.yml
├── pyproject.toml
└── shell.nix            # Окружение NixOS
```
