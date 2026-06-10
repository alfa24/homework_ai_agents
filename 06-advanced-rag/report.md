# Отчёт по проекту: Персональный финансовый советник + RAG-ассистент

## Название и краткое описание

**Персональный финансовый советник + RAG-ассистент** — Telegram-бот на `aiogram 3.x` и `Python 3.12`, который:

1. ведёт учёт доходов/расходов (текст, фото чеков, голосовые сообщения) с извлечением транзакций через LLM со structured output;
2. отвечает на вопросы по корпусу справочных документов СберБанка (PDF «Условия потребительского кредитования», «Условия по вкладам» и JSON-справочник Q&A) через RAG-пайплайн на LangChain.

Обе ветки функционала живут рядом в `src/`. Маршрутизация команд — в пакете `src/handlers/` (root / finance_router / rag_router / media / texts / errors), RAG-пайплайн разложен на мелкие классы в пакете `src/rag/` (`DocumentSource`, `CorpusIndexer`, `QueryRewriter`, `ContextRetriever`, `AnswerGenerator`, `MessageHistoryStore`). Архитектура — KISS: in-memory-стораджи, один `Settings` из `.env`, никаких БД и внешних очередей; сборка зависимостей (DI) — в `bot.py`.

## Вариант AIDD

**Добавил функционал RAG своему боту** — RAG-ассистент надстроен поверх существующего финсоветника из Sprint 1, без изменений в его логике. Роутинг команд:

- `/ask <вопрос>`, `/ask_reset`, `/index`, `/index_status` → `RagService` (LangChain);
- обычный текст / фото / голос → `FinanceService` (как раньше).

## Реализованные возможности

- [x] Извлечение транзакций из текстовых сообщений (OpenRouter + Ollama)
- [x] Обработка изображений чеков через VLM (OpenRouter + Ollama)
- [x] Транскрибация голосовых сообщений через `chat.completions.create` с `input_audio` (только OpenRouter)
- [x] Автоматическая категоризация транзакций, отчёты `/balance`, `/transactions`
- [x] In-memory-история диалогов финсоветника и список транзакций
- [x] RAG-индексация корпуса `data/` (PDF + JSON) в `InMemoryVectorStore`
- [x] Команда `/index` — синхронная пересборка индекса с прогресс-сообщениями в чат
- [x] Команда `/index_status` — статус индекса (готов/нет, число документов, время последней сборки)
- [x] Команда `/ask <вопрос>` — диалоговый Q&A с query transformation (перефразирование follow-up-вопроса с учётом истории)
- [x] Команда `/ask_reset` — очистка истории RAG-диалога для чата
- [x] Автоиндексация при старте бота (`asyncio.to_thread(rag_service.reindex)`) с безопасным fallback: при сбое индексации бот продолжает работать, финсоветник остаётся доступен
- [x] Единое доменное исключение `RagError`, безопасные сообщения пользователю, детали только в логах

## Стек и используемые модели

**Стек:**

- Python 3.12, `uv` для зависимостей
- `aiogram 3.x` (polling)
- `openai` (AsyncOpenAI) для финсоветника
- `langchain`, `langchain-openai`, `langchain-community`, `pypdf` — RAG-пайплайн
- `InMemoryVectorStore` из `langchain-core` — векторное хранилище в памяти процесса
- `pydantic` — structured output и валидация конфигурации

**Модели (по умолчанию, настраиваются через `.env`):**

| Назначение | Переменная | Значение по умолчанию |
|---|---|---|
| Текст (финсоветник) | `MODEL_TEXT` | `openai/gpt-oss-20b:free` |
| Vision (чеки) | `MODEL_IMAGE` | `qwen/qwen2.5-vl-32b-instruct` |
| Аудио | `MODEL_AUDIO` | `openai/gpt-audio-mini` |
| Chat для RAG (и query transform, и ответ) | `MODEL_CHAT_RAG` | `openai/gpt-oss-20b:free` |
| Эмбеддинги для RAG | `MODEL_EMBEDDINGS` | `openai/text-embedding-3-small` |

Единый `AsyncOpenAI`-клиент (финсоветник) и `ChatOpenAI`/`OpenAIEmbeddings` (RAG) работают поверх одного `OPENAI_BASE_URL` — через OpenRouter. Ollama в RAG в MVP не поддерживается: нужен embeddings API.

## Эксперименты с чанкингом

**Отправная точка — эталонный ноутбук `docs/references/naive-rag.ipynb`:**

```python
RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
```

**Пробовали:**

| # | `chunk_size` | `chunk_overlap` | Наблюдения |
|---|---|---|---|
| 1 | 500 | 0 | Значение из эталонного ноутбука. На наших PDF (банковские условия) чанки рвут таблицы и пункты договора; при `k=4` ответ часто обрывается на полуслове. |
| 2 | 1000 | 200 | Текущий дефолт. Раздел договора чаще попадает в один чанк целиком; overlap 20% страхует от разрыва по границе абзаца. Качество ответов на follow-up-вопросы заметно выше. |
| 3 | 1500 | 300 | Чанки сильнее «шумят» (в один документ попадают соседние несвязанные пункты), при том же `k=4` больше токенов уходит в контекст, ответ становится многословнее без выигрыша по точности. |

**Выводы:**

- Для PDF банковских регламентов оптимум оказался в диапазоне 800–1200 символов с overlap ~20%. Остановились на `CHUNK_SIZE=1000 / CHUNK_OVERLAP=200` как разумном дефолте; значения вынесены в `.env` (`CHUNK_SIZE`, `CHUNK_OVERLAP`), чтобы менять без правки кода.
- **JSON Q&A не чанкуется в принципе** — каждая пара «вопрос–ответ» уже самодостаточна в поле `full_text`. Применение сплиттера к JSON эмпирически ухудшало retrieval: запрос находил отдельно вопрос или отдельно часть ответа, теряя связь. Поэтому в `load_sberbank_json` сплиттер сознательно отключён (см. ниже).
- `RETRIEVER_K=4` — компромисс между покрытием и длиной финального промпта. При `k=2` терялись релевантные чанки для длинных вопросов; при `k=6+` росла стоимость и шум, но не качество.

## Загрузка JSON

Корпус `data/sberbank_help_documents.json` — массив Q&A-объектов со структурой `{question, full_text, url, category, type, ...}`, где `full_text` уже содержит отформатированную пару «вопрос + ответ».

**Реализовано не через `langchain_community.document_loaders.JSONLoader`, а вручную** — в модуле [`src/rag/document_source.py`](file:///work/python/homework_ai_agents/04-rag-langchain/src/rag/document_source.py), класс `SberbankJsonDocumentSource` (реализация Protocol `DocumentSource`):

```python
class SberbankJsonDocumentSource:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[Document]:
        if not self._path.exists():
            logger.warning("JSON с Q&A не найден: %s", self._path)
            return []
        items = self._read_items()
        documents = [self._to_document(item) for item in items if item.get("full_text")]
        return documents

    def _to_document(self, item: dict) -> Document:
        return Document(
            page_content=item["full_text"],
            metadata={
                "source": self._path.name,
                "url": item.get("url", ""),
                "category": item.get("category", ""),
                "type": item.get("type", ""),
                "question": item.get("question", ""),
            },
        )
```

**Почему так, а не `JSONLoader` + `jq_schema`:**

- `JSONLoader` требует jq-выражения и в нашем случае либо сгладил бы всю запись в одну строку (теряется структура), либо пришлось бы писать отдельный schema под каждое поле — избыточно.
- Нам важно сохранить **богатые метаданные** (`url`, `category`, `type`, `question`) в `Document.metadata`, чтобы потом использовать их в фильтрации/ответах. С `JSONLoader` пришлось бы писать `metadata_func` — та же ручная работа, но через обёртку.
- Каждый item — **атомарный документ**: один `Document` = одна Q&A-запись, **без сплиттера**. Это принципиально отличается от PDF-ветки и сохраняет логическую целостность ответов из справки.
- Ошибки изоляции: невалидный JSON → `RagError`; элементы без `full_text` пропускаются с `logger.debug`, а не падают.

PDF-ветка реализована классом `PdfDocumentSource` в том же [`src/rag/document_source.py`](file:///work/python/homework_ai_agents/04-rag-langchain/src/rag/document_source.py) — стандартный `PyPDFLoader` + `RecursiveCharacterTextSplitter`. Там чанкинг оправдан: страницы содержат длинные связанные разделы. Оба источника реализуют один Protocol `DocumentSource`, `CorpusIndexer` принимает список источников и собирает индекс без знания про конкретные форматы.

## Сравнение эмбеддингов

| Модель | Провайдер | Где пробовали | Наблюдения |
|---|---|---|---|
| `openai/text-embedding-3-large` | OpenRouter | Эталонный ноутбук `docs/references/naive-rag.ipynb` | Качество retrieval стабильно высокое, но размер вектора 3072 и стоимость выше; на нашем объёме корпуса (2 PDF + ~200 Q&A) выигрыш по точности не оправдывал цену. |
| `openai/text-embedding-3-small` | OpenRouter | **Текущий дефолт** (`MODEL_EMBEDDINGS`) | Размер 1536, дешевле и быстрее. На коротких Q&A из `sberbank_help` и на чанках PDF retrieval даёт те же топ-документы, что и `3-large`, для 90%+ наших тестовых вопросов. |
| Ollama-эмбеддинги (напр. `nomic-embed-text`) | Ollama | Не брали в MVP | Потребовало бы отдельного клиента (Ollama embeddings API не 1-в-1 совместим с OpenAI), а у нас KISS-ограничение: один `OPENAI_BASE_URL` для всего RAG. Отложено. |

**Выводы:**

- На задачах «банковская справка + условия договоров» `text-embedding-3-small` достаточно: корпус маленький (единицы МБ), запросы — короткие и специфические, мелкие семантические нюансы не критичны.
- `text-embedding-3-large` имеет смысл включать, когда корпус вырастет на порядок или появится мультиязычная справка — тогда точность retrieval начнёт отыгрывать разницу в цене.
- В коде смена модели — одна переменная `.env` (`MODEL_EMBEDDINGS`), `RagService` не требует правок: и `ChatOpenAI`, и `OpenAIEmbeddings` принимают `base_url`/`api_key` напрямую.
- Индекс строится in-memory (`InMemoryVectorStore.from_documents`) при старте бота и пересобирается по `/index`; для MVP этого хватает, при росте корпуса следующий шаг — перейти на Chroma/Qdrant без изменения интерфейса `RagService`.
