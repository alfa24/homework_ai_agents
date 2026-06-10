# Персональный финансовый советник + RAG-ассистент

Telegram бот для учета доходов и расходов с интеграцией LLM через OpenRouter или Ollama, плюс RAG-ассистент по корпусу справочных документов СберБанка на LangChain.

## Возможности

- ✅ Извлечение транзакций из текстовых сообщений
- ✅ Обработка изображений чеков и скриншотов
- ✅ Обработка голосовых сообщений (транскрибация через gpt-audio-mini)
- ✅ Автоматическая категоризация транзакций
- ✅ Отчеты о балансе и статистике
- ✅ История всех транзакций
- ✅ Поддержка локальных моделей через Ollama (кроме голоса)
- ✅ RAG-ассистент по корпусу `data/` (PDF + JSON) с диалоговой памятью (`/ask`, `/ask_reset`, `/index`, `/index_status`)

## Установка

### Требования

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) - менеджер зависимостей

### Шаги установки

1. Клонируйте репозиторий:
   ```bash
   git clone <repository-url>
   cd telegram-llm-bot
   ```

2. Установите зависимости:
   ```bash
   make install
   ```

## Конфигурация

### 1. Получите токен Telegram бота

1. Найдите @BotFather в Telegram
2. Отправьте `/newbot` и следуйте инструкциям
3. Скопируйте полученный токен

### 2. Выберите провайдера LLM

#### Вариант A: OpenRouter (облачный)

1. Зарегистрируйтесь на [OpenRouter.ai](https://openrouter.ai/)
2. Перейдите в раздел API Keys
3. Создайте новый ключ

#### Вариант B: Ollama (локальная модель)

1. Установите Ollama:
   ```bash
   # macOS / Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # или через Homebrew на macOS
   brew install ollama
   ```

2. Запустите Ollama сервер:
   ```bash
   ollama serve
   ```

3. Установите модель (в отдельном терминале):
   ```bash
   # Для текстовых сообщений (с поддержкой structured output)
   ollama pull llama3.2
   
   # Для изображений (опционально, для итерации 5)
   ollama pull llama3.2-vision
   ```

### 3. Настройте переменные окружения

1. Скопируйте пример конфигурации:
   ```bash
   cp .env.example .env
   ```

2. Отредактируйте `.env` файл:

   **Для OpenRouter:**
   ```bash
   TELEGRAM_TOKEN=ваш_токен_от_BotFather
   OPENAI_API_KEY=ваш_ключ_от_OpenRouter
   OPENAI_BASE_URL=https://openrouter.ai/api/v1
   MODEL_TEXT=openai/gpt-oss-20b:free
   MODEL_IMAGE=qwen/qwen2.5-vl-32b-instruct
   MODEL_AUDIO=openai/gpt-audio-mini
   # RAG (требует OpenRouter/OpenAI — Ollama для RAG не поддерживается в MVP)
   MODEL_CHAT_RAG=openai/gpt-4o-mini
   MODEL_EMBEDDINGS=openai/text-embedding-3-small
   DATA_DIR=data
   RETRIEVER_K=4
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```

   **Для Ollama:**
   ```bash
   TELEGRAM_TOKEN=ваш_токен_от_BotFather
   OPENAI_API_KEY=ollama
   OPENAI_BASE_URL=http://localhost:11434/v1
   MODEL_TEXT=gpt-oss:20b
   MODEL_IMAGE=qwen3-vl:8b-instruct
   # MODEL_AUDIO не используется: Ollama не поддерживает audio input.
   # Голосовые сообщения доступны только через OpenRouter/OpenAI.
   # RAG также недоступен через Ollama в MVP: требуется OpenAI-совместимый провайдер с embeddings API.
   ```

**Описание переменных:**

- `TELEGRAM_TOKEN` - токен бота от @BotFather (обязательно)
- `OPENAI_API_KEY` - API ключ от OpenRouter или любое значение для Ollama (обязательно)
- `OPENAI_BASE_URL` - URL API провайдера (по умолчанию: https://openrouter.ai/api/v1)
- `MODEL_TEXT` - модель для обработки текстовых сообщений
- `MODEL_IMAGE` - модель для обработки изображений (должна поддерживать vision)
- `MODEL_AUDIO` - модель для транскрибации голосовых сообщений (работает только через провайдеры с поддержкой audio input)
- `PROXY_URL` - SOCKS5 прокси для Telegram API (опционально, пример: `socks5://user:password@host:port`)
- `SYSTEM_PROMPT_PATH` - путь к файлу с системным промптом (по умолчанию: `prompts/system_prompt.txt`)
- `SYSTEM_PROMPT` - альтернатива: системный промпт напрямую в переменной окружения (опционально)
- `MODEL_CHAT_RAG` - модель для RAG (query transform + генерация ответа), требует OpenRouter/OpenAI
- `MODEL_EMBEDDINGS` - модель эмбеддингов для векторного индекса (OpenRouter/OpenAI)
- `DATA_DIR` - каталог с корпусом для RAG (по умолчанию: `data`)
- `RETRIEVER_K` - число извлекаемых чанков на запрос (по умолчанию: `4`)
- `CHUNK_SIZE` / `CHUNK_OVERLAP` - параметры разбиения PDF на чанки (по умолчанию: `1000` / `200`)
- `SHOW_SOURCES` - показывать источники в ответе `/ask` (по умолчанию: `false`)
- `LANGCHAIN_TRACING_V2` - включить трейсинг LangSmith (`true` / не задано, опционально)
- `LANGCHAIN_API_KEY` - API-ключ LangSmith (опционально)
- `LANGCHAIN_PROJECT` - имя проекта в LangSmith (по умолчанию: `default`)

## LangSmith-трейсинг

LangChain автоматически отправляет трейсы всех RAG-запросов при наличии переменных окружения:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=05-monitoring-qa
```

**Настройка:**
1. Зарегистрируйтесь на [smith.langchain.com](https://smith.langchain.com)
2. Создайте API-ключ в настройках
3. Добавьте переменные в `.env`
4. Запустите бота и выполните `/ask` — трейс появится в дашборде LangSmith

Без этих переменных бот работает как обычно, без ошибок.

## Advanced RAG

Бот поддерживает три режима поиска документов, задаваемых переменной `RAG_RETRIEVAL_MODE`:

| Режим | Описание |
|-------|----------|
| `semantic` | Только векторный поиск (по умолчанию) |
| `hybrid` | Semantic + BM25 через `EnsembleRetriever` |
| `hybrid_rerank` | Hybrid + Cross-encoder reranking |

### Провайдеры embeddings

| Переменная | Значения | Описание |
|-----------|----------|----------|
| `EMBEDDINGS_PROVIDER` | `openai`, `huggingface` | Провайдер embeddings для RAG |
| `RAGAS_EMBEDDINGS_PROVIDER` | `openai`, `huggingface` | Провайдер embeddings для RAGAS evaluation |

### Ключевые переменные Advanced RAG

```bash
RAG_RETRIEVAL_MODE=semantic          # semantic | hybrid | hybrid_rerank
EMBEDDINGS_PROVIDER=openai           # openai | huggingface
MODEL_EMBEDDINGS=openai/text-embedding-3-small
SEMANTIC_RETRIEVER_K=4
BM25_RETRIEVER_K=8
HYBRID_RETRIEVER_K=8
HYBRID_SEMANTIC_WEIGHT=0.5
HYBRID_BM25_WEIGHT=0.5
MODEL_CROSS_ENCODER=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
RERANKER_TOP_K=4
```

## Запуск

```bash
make run
```

Бот запустится в режиме polling и будет работать до остановки (Ctrl+C).

## Использование

### Команды бота

**Финансовый советник:**
- `/start` - начать новый диалог (сбрасывает историю и транзакции)
- `/balance` - показать баланс, доходы, расходы и статистику по категориям
- `/transactions` - показать список всех транзакций

**RAG-ассистент (справка СберБанка):**
- `/ask <вопрос>` - задать вопрос по корпусу документов (PDF + JSON в `data/`); учитывает историю диалога
- `/ask_reset` - очистить историю RAG-диалога для текущего чата
- `/index` - вручную пересобрать векторный индекс из `data/`
- `/index_status` - показать состояние индекса (построен/нет, число документов, время последней сборки)

> Корпус: `data/ouk_potrebitelskiy_kredit_lph.pdf`, `data/usl_r_vkladov.pdf`, `data/sberbank_help_documents.json`. При старте бот автоматически выполняет индексацию; если она не удалась — бот продолжит работу, и `/ask` будет доступен после ручного `/index`. Финансовый советник работает независимо от RAG.

> RAG требует OpenRouter/OpenAI-совместимого провайдера (нужен embeddings API). Через Ollama RAG в MVP не поддерживается — финсоветник на Ollama работает как раньше, RAG-команды в этом режиме будут возвращать ошибку.

### Примеры использования

**Пример 1: Текстовая транзакция**
```
Пользователь: Сегодня купил продукты на 1500 рублей
Бот: Записал ваш расход на продукты в размере 1500 рублей.

✅ Найдено и сохранено 1 транзакция
💵 Баланс: -1500 руб.
```

**Пример 2: Обработка чека**
```
Пользователь: [отправляет фото чека]
Бот: Обработал чек. Найдено 3 позиции на сумму 2340 рублей.

✅ Найдено и сохранено 1 транзакция
💵 Баланс: -3840 руб.
```

**Пример 3: Голосовое сообщение**
```
Пользователь: [отправляет голосовое сообщение "Потратил 500 рублей на кофе"]
Бот: 🎤 Распознано: Потратил 500 рублей на кофе
Бот: Записал ваш расход на кофе в размере 500 рублей.

✅ Найдено и сохранено 1 транзакция
💵 Баланс: -500 руб.
```

**Пример 4: Проверка баланса**
```
Пользователь: /balance
Бот: 💵 Отчет о балансе

📊 Баланс: -1500.00 руб.
💰 Доходы: 0.00 руб.
💸 Расходы: 1500.00 руб.

📈 Всего транзакций: 1

**Статистика по категориям:**
💸 продукты: -1500.00 руб.
```

## Ограничения

- Максимальная длина текстового сообщения: 4000 символов
- История диалога и транзакции хранятся в памяти (при перезапуске бота теряются)
- Бот обрабатывает текст, изображения и голосовые сообщения (не PDF, не видео, не файлы других форматов)
- Для обработки изображений требуется модель с поддержкой vision
- Голосовые сообщения обрабатываются через модели с поддержкой audio input (gpt-audio-mini, gpt-audio). Ollama не поддерживает транскрибацию аудио.

## Разработка

### Структура проекта

```
├── src/
│   ├── audio_converter.py       # Конвертация аудио (OGG → WAV)
│   ├── bot.py                   # Точка входа, инициализация, автоиндексация RAG
│   ├── config.py                # Загрузка конфигурации
│   ├── conversation_store.py    # История диалогов финсоветника
│   ├── document_loader.py       # Загрузка PDF и JSON-корпуса для RAG
│   ├── exceptions.py            # Доменные исключения (включая RagError)
│   ├── finance_service.py       # Бизнес-логика (извлечение, баланс)
│   ├── handlers.py              # Обработчики сообщений и команд
│   ├── llm_client.py            # Интеграция с LLM
│   ├── models.py                # Pydantic модели для транзакций
│   ├── rag_conversation_store.py# История RAG-диалога (in-memory буфер BaseMessage)
│   ├── rag_service.py           # RAG: индексация, query transform, retrieval, ответ
│   ├── report_formatter.py      # Форматирование отчётов
│   └── transaction_store.py     # Хранение транзакций
├── data/                        # Корпус RAG (PDF + JSON справки СберБанка)
├── prompts/
│   └── system_prompt.txt        # Единый системный промпт финсоветника
├── .env                         # Конфигурация (не коммитится)
├── .env.example                 # Пример конфигурации
├── Makefile                     # Команды для работы
├── pyproject.toml               # Зависимости проекта
└── README.md                    # Документация
```

### Команды Makefile

- `make install` - установить зависимости
- `make run` - запустить бота

## Лицензия

MIT

