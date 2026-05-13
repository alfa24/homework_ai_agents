# Техническое видение проекта

## Технологии

**Основные технологии:**
- **Python 3.12** - основной язык разработки
- **uv** - управление зависимостями и виртуальным окружением
- **aiogram 3.x** - фреймворк для Telegram Bot API (polling)
- **openai** - клиент для работы с LLM через OpenRouter/Ollama (единый интерфейс, включая audio input через `chat.completions.create` для транскрибации)
- **pydantic** - валидация данных и structured output для LLM
- **python-dotenv** - для работы с переменными окружения
- **Make** - автоматизация сборки и запуска
- **langchain / langchain-openai / langchain-community** - RAG-пайплайн для справочного ассистента (query transformation, retriever, промпт-цепочки)
- **pypdf** - загрузка PDF-документов из `data/` в `Document`
- **InMemoryVectorStore** (из `langchain-core`) - векторное хранилище в памяти процесса, без внешних БД

## Принципы разработки

**Принципы:**
- **KISS** (Keep It Simple, Stupid) - максимальная простота решений
- **YAGNI** (You Aren't Gonna Need It) - реализуем только то, что нужно сейчас
- **Монолитная архитектура** - весь код в одном месте, никаких микросервисов
- **Прямолинейный код** - минимум абстракций, максимум читаемости
- **Быстрый старт** - от идеи до рабочего прототипа за минимальное время

**Что НЕ делаем:**
- Не создаем сложные архитектурные паттерны
- Не делаем преждевременную оптимизацию
- Не добавляем функции "на будущее"
- Не усложняем без крайней необходимости

## Структура проекта

```
/
├── src/
│   ├── audio_converter.py         # Конвертация аудио (OGG → WAV)
│   ├── bot.py                     # Основной файл бота, инициализация aiogram
│   ├── config.py                  # Загрузка конфигурации из .env (Settings dataclass)
│   ├── conversation_store.py      # Хранение истории диалогов финсоветника в памяти
│   ├── document_loader.py         # Загрузка PDF и JSON из data/ в LangChain Document'ы
│   ├── exceptions.py              # Доменные исключения (LLMError, AudioError, ConfigError, RagError)
│   ├── finance_service.py         # Бизнес-логика финсоветника (извлечение транзакций, баланс)
│   ├── handlers.py                # Обработчики команд и сообщений Telegram (финсоветник + RAG)
│   ├── llm_client.py              # Работа с LLM через OpenRouter/Ollama (финсоветник)
│   ├── models.py                  # Pydantic модели для транзакций
│   ├── rag_conversation_store.py  # Хранение истории RAG-диалога в памяти
│   ├── rag_service.py             # RAG-сервис: InMemoryVectorStore + query transform + retriever
│   ├── report_formatter.py        # Форматирование отчётов (баланс, транзакции)
│   └── transaction_store.py       # Хранение транзакций в памяти
├── prompts/
│   └── system_prompt.txt          # Единый системный промпт финсоветника (текст + изображения)
├── data/
│   ├── ouk_potrebitelskiy_kredit_lph.pdf  # PDF: условия потребительского кредитования
│   ├── usl_r_vkladov.pdf                  # PDF: условия по вкладам
│   └── sberbank_help_documents.json       # 212 Q&A справки по картам
├── .env                      # Переменные окружения (токены, настройки)
├── .env.example              # Пример конфигурации
├── pyproject.toml            # Конфигурация проекта для uv
├── Makefile                  # Команды для запуска и управления
└── README.md                 # Документация по запуску
```

**Принцип:** Все Python-файлы в одной папке `src/`. Каждый класс — отдельный файл. Простой OOP без сложной иерархии. Компоненты финсоветника и RAG-ассистента живут рядом и не пересекаются: общий только `handlers.py`, который маршрутизирует команды.

## Архитектура проекта

**Компоненты:**

1. **bot.py** - точка входа
   - Инициализирует aiogram Bot и Dispatcher
   - Регистрирует handlers
   - Запускает polling

2. **handlers.py** - обработка событий
   - `/start` - приветствие и очистка истории/транзакций
   - `/balance` - отчет о балансе и статистике
   - `/ask <вопрос>` - RAG-ответ по корпусу документов (`RagService.answer`)
   - `/ask_reset` - очистка истории RAG-диалога для текущего чата
   - `/index` - пересборка векторного индекса RAG (`RagService.reindex`)
   - `/index_status` - статус индекса (`RagService.status`)
   - Обработчик текстовых сообщений → извлечение транзакций через LLM → сохранение транзакций → показ ответа + статус + баланс
   - Обработчик изображений → извлечение транзакций через VLM → сохранение транзакций → показ ответа + статус + баланс
   - Обработчик голосовых сообщений → скачивание ogg → конвертация в WAV → транскрибация через chat.completions.create с input_audio → распознанный текст передаётся в существующий текстовый пайплайн
   - Хранит историю диалогов финсоветника в памяти: `ConversationStore` (chat_id → список сообщений)
   - Хранит транзакции в памяти: `TransactionStore` (chat_id → список транзакций)
   - Хранит историю RAG-диалога в памяти: `RagConversationStore` (chat_id → список `BaseMessage`)

6. **rag_service.py** - RAG-ассистент
   - Владеет `InMemoryVectorStore` и цепочками LangChain
   - `reindex()` - загружает документы через `document_loader`, строит индекс заново
   - `answer(chat_id, question)` - query transformation с учётом истории → retriever top-K → финальный ответ LLM по контексту
   - `status()` - количество документов/чанков и время последней индексации
   - Использует `ChatOpenAI` и `OpenAIEmbeddings` поверх OpenRouter (через `base_url`)

7. **document_loader.py** - загрузка корпуса
   - `load_pdfs(dir)` - PDF-файлы через `PyPDFLoader` + `RecursiveCharacterTextSplitter` (`CHUNK_SIZE`, `CHUNK_OVERLAP`)
   - `load_sberbank_json(path)` - каждый Q&A-элемент становится одним `Document` (`page_content=full_text`), без сплиттера; метаданные `source`, `url`, `category`, `question`

3. **llm_client.py** - интеграция с LLM
   - Метод `extract_from_text()` - обработка текстовых сообщений со structured output
   - Метод `extract_from_image()` - обработка изображений (VLM) со structured output
   - Метод `transcribe_audio()` - транскрибация аудио через `chat.completions.create` с `input_audio`
   - Единый интерфейс через AsyncOpenAI для OpenRouter и Ollama
   - Переключение между внешними и локальными моделями через конфигурацию

4. **models.py** - модели данных
   - Pydantic модели для транзакций (Transaction, TransactionResponse)
   - Enums для типов транзакций (TransactionType, TransactionFrequency)
   - Валидация данных транзакций

5. **config.py** - конфигурация
   - Иммутабельный `Settings` dataclass (`frozen=True, slots=True`) с fail-fast валидацией
   - Поля: `telegram_token`, `openai_api_key`, `openai_base_url`, `model_text`, `model_image`, `model_audio`, `system_prompt`, `proxy_url`, `model_chat_rag`, `model_embeddings`, `data_dir`, `retriever_k`, `chunk_size`, `chunk_overlap`
   - Единый системный промпт (`system_prompt`) — загружается из файла (`SYSTEM_PROMPT_PATH`) или переменной окружения (`SYSTEM_PROMPT`)
   - `_require()` для обязательных параметров, `_load_prompt()` для гибкой загрузки промпта
   - `model_text` — модель для текстовых сообщений, `model_image` — модель для изображений (vision), `model_audio` — модель для транскрибации

6. **rag_service.py** - RAG-ассистент
   - Владеет `InMemoryVectorStore` и цепочками LangChain
   - `reindex()` - загружает документы через `document_loader`, строит индекс заново
   - `answer(chat_id, question)` - query transformation с учётом истории → retriever top-K → финальный ответ LLM по контексту
   - `status()` - количество документов/чанков и время последней индексации
   - Использует `ChatOpenAI` и `OpenAIEmbeddings` поверх OpenRouter (через `base_url`)

7. **document_loader.py** - загрузка корпуса
   - `load_pdfs(dir)` - PDF-файлы через `PyPDFLoader` + `RecursiveCharacterTextSplitter` (`CHUNK_SIZE`, `CHUNK_OVERLAP`)
   - `load_sberbank_json(path)` - каждый Q&A-элемент становится одним `Document` (`page_content=full_text`), без сплиттера; метаданные `source`, `url`, `category`, `question`

**Поток данных (текстовые сообщения):**
```
Telegram → handlers.py (последнее сообщение) → finance_service.py (structured output) → llm_client.py → OpenRouter/Ollama → 
finance_service.py → handlers.py (извлечь транзакции, сохранить в transactions, показать ответ + статус + баланс) → Telegram
```

**Поток данных (изображения):**
```
Telegram → handlers.py (изображение → base64) → finance_service.py (VLM + structured output) → llm_client.py → OpenRouter/Ollama → 
finance_service.py → handlers.py (извлечь транзакции, сохранить в transactions, показать ответ + статус + баланс) → Telegram
```

**Поток данных (голосовые сообщения):**
```
Telegram → handlers.py (скачать ogg, конвертировать в WAV) → llm_client.py.transcribe_audio() → OpenRouter (chat.completions.create с input_audio) → текст →
handlers.py (показать распознанный текст) → finance_service.process_text_message() → извлечь транзакции + ответ + баланс → Telegram
```

**Поток данных (RAG, `/ask <вопрос>`):**
```
Telegram → handlers.py (команда /ask) → RagService.answer(chat_id, question) →
  rag_query_transform_chain (LLM переписывает вопрос с учётом истории) →
  InMemoryVectorStore.as_retriever(k=RETRIEVER_K) → top-K Document'ов →
  LLM (ChatOpenAI) генерирует ответ по контексту → handlers.py → Telegram
```

**Поток данных (индексация, `/index` и старт):**
```
bot.py (startup) / handlers.py (/index) → RagService.reindex() →
  document_loader.load_pdfs(DATA_DIR) + load_sberbank_json(...) → list[Document] →
  OpenAIEmbeddings → InMemoryVectorStore.from_documents(...) → готовый индекс в памяти
```

**Принцип:** Простой OOP с внедрением зависимостей через конструктор. Финсоветник: Handler → FinanceService → LLMClient. RAG: Handler → RagService (LangChain chain + retriever + vector store). Две ветки независимы.

## Модель данных

**Хранение в памяти (без БД):**

Отдельные классы-хранилища в `src/`:
- `ConversationStore` — история диалогов: `dict[int, list[dict]]` (chat_id → список сообщений)
- `TransactionStore` — транзакции пользователей: `dict[int, list[Transaction]]` (chat_id → список транзакций)

Хранилища внедряются через конструктор в `FinanceService` и `handlers`.

**Структура истории диалога:**
```python
chat_conversations[chat_id] = [
    {"role": "system", "content": "системный промпт"},
    {"role": "user", "content": "сообщение пользователя"},
    {"role": "assistant", "content": "ответ LLM"},
    ...
]
```

**Структура транзакций:**
```python
from models import Transaction, TransactionType, TransactionFrequency

transactions[chat_id] = [
    Transaction(
        date=date(2024, 1, 15),
        time=time(14, 30),
        type=TransactionType.EXPENSE,
        amount=1500.0,
        frequency=TransactionFrequency.DAILY,
        category="продукты",
        description="Молоко, хлеб, яйца в магазине Пятёрочка"
    ),
    ...
]
```

**Pydantic модели (src/models.py):**

```python
from pydantic import BaseModel, Field
from datetime import date, time
from enum import Enum

class TransactionType(str, Enum):
    INCOME = "income"      # доход
    EXPENSE = "expense"    # расход

class TransactionFrequency(str, Enum):
    DAILY = "daily"           # повседневные
    PERIODIC = "periodic"     # периодические
    ONE_TIME = "one_time"     # разовые

class Transaction(BaseModel):
    date: date                           # дата транзакции
    time: time | None = None            # время (опционально)
    type: TransactionType                # доход/расход
    amount: float = Field(gt=0)          # сумма (строго положительная)
    frequency: TransactionFrequency       # тип (повседневные, периодические, разовые)
    category: str                        # категория (продукты, рестораны, такси и т.д.)
    description: str = ""                # описание транзакции (подробная информация о товарах, услугах, источнике, контрагенте и т.п.)

class TransactionResponse(BaseModel):
    transactions: list[Transaction] = []  # список транзакций (может быть пустым)
    answer: str                           # текстовый ответ пользователю (обязателен)
```

**Категории расходов/доходов:**
- Базовый список: продукты, рестораны, такси, транспорт, образование, путешествия, развлечения, здоровье, одежда, другие
- LLM может предлагать новые категории, которые добавляются в список
- Если категория не определена - используется "другие"

**Операции:**
- При `/start` - очищаем историю и транзакции для данного чата
- При новом сообщении - извлекаем транзакции ТОЛЬКО из последнего сообщения
- Сохраняем транзакции в `transactions[chat_id]`
- При перезапуске бота - вся история и транзакции теряются

**Принцип:** Максимальная простота. Никаких БД, файлов, сериализации. Все данные живут только в runtime.

## Работа с LLM

**Используемая библиотека:** `openai` (официальный Python client, асинхронная версия)

**Настройка:**
```python
from openai import AsyncOpenAI

# Единый интерфейс для OpenRouter и Ollama
# Переключение моделей через изменение base_url и model в .env
# LLMClient получает Settings через конструктор
client = AsyncOpenAI(
    api_key=settings.openai_api_key,  # для Ollama можно использовать любое значение
    base_url=settings.openai_base_url  # https://openrouter.ai/api/v1 или http://localhost:11434/v1
)
```

**Методы в llm_client.py (класс LLMClient):**

**1. Обработка текстовых сообщений:**
```python
async def extract_from_text(
    self,
    last_message: str,
    message_history: list[dict]
) -> TransactionResponse:
    # Structured output через response_format с JSON schema из Pydantic
    response = await self._client.chat.completions.create(
        model=self._settings.model_text,
        messages=[
            {"role": "system", "content": self._settings.system_prompt},
            *message_history[-10:],  # последние 10 сообщений для контекста
            {"role": "user", "content": last_message}
        ],
        response_format={"type": "json_schema", "json_schema": {
            "name": "transaction_response",
            "schema": TransactionResponse.model_json_schema(),
            "strict": True
        }}
    )
    # Парсинг JSON ответа в TransactionResponse
    return TransactionResponse.model_validate_json(response.choices[0].message.content)
```

**2. Обработка изображений (VLM):**
```python
async def extract_from_image(
    self,
    image_base64: str,
    message_history: list[dict]
) -> TransactionResponse:
    # Vision API с structured output
    response = await self._client.chat.completions.create(
        model=self._settings.model_image,
        messages=[
            {"role": "system", "content": self._settings.system_prompt},
            *message_history[-10:],
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": "Извлеки транзакции из этого изображения"}
                ]
            }
        ],
        response_format={"type": "json_schema", "json_schema": {
            "name": "transaction_response",
            "schema": TransactionResponse.model_json_schema(),
            "strict": True
        }}
    )
    return TransactionResponse.model_validate_json(response.choices[0].message.content)
```

**3. Транскрибация голосовых сообщений (audio input):**
```python
async def transcribe_audio(
    self,
    audio_bytes: bytes,
    filename: str = "voice.wav"
) -> str:
    # Транскрибация через chat.completions.create с input_audio
    audio_base64 = base64.b64encode(audio_bytes).decode()
    response = await self._client.chat.completions.create(
        model=self._settings.model_audio,
        messages=[
            {"role": "user", "content": [
                {"type": "input_audio", "input_audio": {
                    "data": audio_base64,
                    "format": "wav"
                }},
                {"type": "text", "text": "Транскрибируй это аудио"}
            ]}
        ]
    )
    return response.choices[0].message.content
```

**Важные особенности:**
- **Structured output**: Использование Pydantic моделей для валидации ответов LLM через `response_format` с JSON schema
- **Извлечение транзакций**: ТОЛЬКО из последнего сообщения пользователя (не из всей истории) - подчеркнуто в системном промпте
- **Переключение моделей**: Единый интерфейс через AsyncOpenAI, переключение через изменение `OPENAI_BASE_URL` и переменных `MODEL_*` в .env файле

**Параметры из .env:**
- `OPENAI_API_KEY` - ключ от OpenRouter (для Ollama можно любое значение)
- `OPENAI_BASE_URL` - URL API провайдера:
  - Для OpenRouter: `https://openrouter.ai/api/v1`
  - Для Ollama: `http://localhost:11434/v1`
- `MODEL_TEXT` - модель для обработки текстовых сообщений (например `openai/gpt-oss-20b:free` для OpenRouter или `gpt-oss:20b` для Ollama)
- `MODEL_IMAGE` - модель для обработки изображений, должна поддерживать vision (например `qwen/qwen2.5-vl-32b-instruct` для OpenRouter или `qwen3-vl:8b-instruct` для Ollama)
- `MODEL_AUDIO` - модель для транскрибации голосовых сообщений (например `openai/gpt-audio-mini`; доступна только на OpenAI-совместимых провайдерах, таких как OpenRouter)
- `SYSTEM_PROMPT_PATH` - путь к файлу с единым системным промптом (по умолчанию: `prompts/system_prompt.txt`)
- `SYSTEM_PROMPT` - альтернатива: системный промпт напрямую (если указан, используется вместо файла)
- `PROXY_URL` - SOCKS5 прокси для Telegram API (опционально)

**Переключение между провайдерами:**
Для использования OpenRouter:
```bash
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL_TEXT=openai/gpt-oss-20b:free
MODEL_IMAGE=qwen/qwen2.5-vl-32b-instruct
MODEL_AUDIO=openai/gpt-audio-mini
```

Для использования Ollama:
```bash
OPENAI_BASE_URL=http://localhost:11434/v1
MODEL_TEXT=gpt-oss:20b
MODEL_IMAGE=qwen3-vl:8b-instruct
# MODEL_AUDIO не используется: Ollama не поддерживает audio input.
# Обработка голосовых сообщений доступна только при работе через OpenRouter/OpenAI.
```

**Обработка ошибок:**
- try/except для сетевых ошибок
- Возврат простого сообщения об ошибке пользователю
- Валидация ответов через Pydantic (автоматическая обработка ошибок парсинга)

**Принцип:** Асинхронный запрос-ответ с structured output. Никакого retry, никаких очередей, никакого streaming.

### RAG-ассистент (LangChain)

RAG-ассистент использует LangChain поверх того же OpenRouter: `ChatOpenAI` и `OpenAIEmbeddings` с явным `base_url=OPENAI_BASE_URL` и `api_key=OPENAI_API_KEY`. Отдельного клиента не создаём — LangChain внутри работает с официальным `openai` SDK.

**Индексация корпуса (`data/`):**
- PDF: `PyPDFLoader` + `RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)` (по умолчанию 1000/200).
- `sberbank_help_documents.json`: каждый элемент — самодостаточный Q&A, уже отформатированный в поле `full_text`. Сплиттер не применяем, чтобы не разрывать пару «вопрос-ответ». Каждый элемент → один `Document` c `page_content=full_text` и `metadata={source: "sberbank_help", url, category, question}`.
- Итоговый список `Document` кладётся в `InMemoryVectorStore.from_documents(docs, embedding=OpenAIEmbeddings(...))`.

**Пайплайн ответа (`RagService.answer`):**
1. `rag_query_transform_chain` (см. `docs/references/naive-rag.ipynb`): LLM берёт историю диалога (`list[BaseMessage]` из `RagConversationStore`) и текущий вопрос, переписывает его в самодостаточный поисковый запрос.
2. `vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})` возвращает top-K документов.
3. Финальная цепочка передаёт найденные фрагменты вместе с исходным вопросом в `ChatOpenAI` (модель `MODEL_CHAT_RAG`) — та же модель используется и для query transformation (KISS: один параметр в конфиге).
4. Ответ добавляется в `RagConversationStore` как `AIMessage`; вопрос — как `HumanMessage`.

**Команды и состояние:**
- `/index` — синхронный reindex с сообщениями о прогрессе в чат (загрузка, разбиение, эмбеддинги, готово).
- `/index_status` — показывает число документов/чанков и время последней индексации.
- `/ask <вопрос>` — обязательный префикс. Без префикса сообщение идёт в финсоветник. История каждого чата хранится отдельно.
- `/ask_reset` — очищает `RagConversationStore[chat_id]`.

**Обработка ошибок RAG:**
- Единое доменное исключение `RagError` (индекс не построен, ошибка LLM/эмбеддингов, пустой корпус).
- Пользователь получает безопасное сообщение; детали — только в логах.

## Системный промпт

Промпт хранится в файле `prompts/system_prompt.txt` — единый для текста и изображений (DRY). Путь к файлу можно настроить через переменную окружения `SYSTEM_PROMPT_PATH`, либо задать промпт напрямую через `SYSTEM_PROMPT`.

Полный текст промпта — см. `prompts/system_prompt.txt`.

## Сценарии работы

**Сценарий 1: Первый запуск**
1. Пользователь отправляет `/start`
2. Бот отвечает приветственным сообщением
3. История диалога инициализируется с системным промптом
4. Список транзакций для пользователя очищается

**Сценарий 2: Обработка текстового сообщения с транзакцией**
1. Пользователь пишет: "Сегодня купил продукты на 1500 рублей"
2. Бот отправляет только последнее сообщение в LLM со structured output
3. LLM извлекает транзакцию и возвращает TransactionResponse
4. Бот сохраняет транзакцию в `transactions[chat_id]`
5. Бот рассчитывает баланс
6. Бот отправляет пользователю:
   - Ответ LLM из поля `answer`
   - Статус: "Найдено и сохранено 1 транзакция"
   - Текущий баланс: "Баланс: -1500 руб."

**Сценарий 3: Обработка изображения (чека)**
1. Пользователь отправляет изображение чека
2. Бот конвертирует изображение в base64
3. Бот отправляет изображение в VLM со structured output
4. VLM распознает текст и извлекает транзакции
5. Бот сохраняет транзакции в `transactions[chat_id]`
6. Бот рассчитывает баланс
7. Бот отправляет пользователю ответ + статус + баланс

**Сценарий 4: Запрос баланса**
1. Пользователь отправляет `/balance`
2. Бот рассчитывает из `transactions[chat_id]`:
   - Баланс = сумма доходов - сумма расходов
   - Общая сумма доходов
   - Общая сумма расходов
   - Статистика по категориям за все время
3. Бот отправляет форматированный отчет

**Сценарий 5: Сброс контекста**
1. Пользователь отправляет `/start`
2. История диалога и транзакции очищаются
3. Начинается новый диалог

**Сценарий 6: Обработка голосового сообщения**
1. Пользователь отправляет голосовое сообщение (ogg/opus)
2. Бот скачивает аудиофайл через Telegram API
3. `audio_converter` конвертирует аудио в WAV (через ffmpeg)
4. Бот вызывает `llm_client.transcribe_audio()` → `chat.completions.create` с `input_audio` возвращает распознанный текст
5. Бот показывает пользователю распознанный текст (для прозрачности)
6. Далее используется существующий текстовый пайплайн: `finance_service.process_text_message()` → извлечение транзакций + ответ + баланс (как в Сценарии 2)
7. Если провайдер — Ollama, бот отвечает пользователю, что голосовые сообщения не поддерживаются

**Ограничения:**
- Бот обрабатывает текст, изображения и голосовые сообщения (не PDF, не видео, не файлы других форматов)
- Голосовые сообщения работают только с OpenAI-совместимыми провайдерами аудио (OpenRouter); Ollama не поддерживается
- Один пользователь не блокирует других (асинхронность)
- При перезапуске бота все истории и транзакции теряются

## Подход к конфигурированию

**Файл .env** (не коммитится в git):
```bash
TELEGRAM_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openrouter_api_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL_TEXT=openai/gpt-oss-20b:free
MODEL_IMAGE=qwen/qwen2.5-vl-32b-instruct
MODEL_AUDIO=openai/gpt-audio-mini
SYSTEM_PROMPT_PATH=prompts/system_prompt.txt

# RAG-ассистент
MODEL_CHAT_RAG=openai/gpt-oss-20b:free
MODEL_EMBEDDINGS=openai/text-embedding-3-small
DATA_DIR=data
RETRIEVER_K=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

**Файл .env.example** (коммитится):
```bash
TELEGRAM_TOKEN=
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL_TEXT=openai/gpt-oss-20b:free
MODEL_IMAGE=qwen/qwen2.5-vl-32b-instruct
MODEL_AUDIO=openai/gpt-audio-mini

# System Prompt
SYSTEM_PROMPT_PATH=prompts/system_prompt.txt
# Альтернативно: можно переопределить промпт напрямую
# SYSTEM_PROMPT=

# SOCKS5 Proxy (опционально)
# PROXY_URL=socks5://user:password@host:port

# RAG-ассистент (Sprint 2)
MODEL_CHAT_RAG=openai/gpt-oss-20b:free
MODEL_EMBEDDINGS=openai/text-embedding-3-small
DATA_DIR=data
RETRIEVER_K=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

**config.py:**
```python
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent


def _load_prompt(file_path: str, env_var: str | None = None) -> str:
    """Возвращает текст промпта: сначала из переменной окружения, затем из файла."""
    if env_var:
        value = os.getenv(env_var)
        if value:
            return value
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    telegram_token: str
    openai_api_key: str
    openai_base_url: str
    model_text: str
    model_image: str | None
    model_audio: str
    system_prompt: str
    proxy_url: str | None

    @classmethod
    def load(cls) -> "Settings":
        model_text = os.getenv("MODEL_TEXT") or os.getenv("MODEL")
        if not model_text:
            raise ConfigError("Missing required environment variable: MODEL_TEXT")

        return cls(
            telegram_token=_require("TELEGRAM_TOKEN"),
            openai_api_key=_require("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            model_text=model_text,
            model_image=os.getenv("MODEL_IMAGE"),
            model_audio=os.getenv("MODEL_AUDIO", "openai/whisper-1"),
            system_prompt=_load_prompt(
                os.getenv("SYSTEM_PROMPT_PATH", "prompts/system_prompt.txt"),
                "SYSTEM_PROMPT",
            ),
            proxy_url=os.getenv("PROXY_URL"),
        )
```

**Принцип загрузки промпта:**
1. Приоритет 1: Переменная окружения `SYSTEM_PROMPT` (если указана напрямую)
2. Приоритет 2: Файл по пути из `SYSTEM_PROMPT_PATH`
3. По умолчанию: `prompts/system_prompt.txt`

**Принципы:**
- Все секреты только в .env
- Нет YAML, JSON, TOML конфигов
- Нет окружений (dev/prod)
- Fail-fast валидация на старте: `_require()` бросает `ConfigError` если обязательная переменная не задана

## Подход к логгированию

**Используем встроенный logging Python:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

**Что логируем:**
- Старт/остановка бота
- Входящие сообщения от пользователей (chat_id + текст)
- Ответы LLM (содержимое ответа + извлеченные транзакции)
- Ошибки при вызове LLM
- Исключения

**Что НЕ логируем:**
- Детальные трейсы успешных операций
- Метрики, аналитика

**Вывод:** Только в stdout/stderr (консоль)

**Принципы:**
- Без внешних библиотек (structlog и т.п.)
- Без файлов, ротации логов
- Без отправки в внешние системы
- Простой текстовый формат


