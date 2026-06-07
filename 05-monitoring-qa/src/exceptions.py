"""Доменные исключения приложения."""


class ConfigError(Exception):
    """Ошибка загрузки/валидации конфигурации."""


class LLMError(Exception):
    """Базовая ошибка LLM-взаимодействия."""


class LLMParseError(LLMError):
    """LLM вернула невалидный или пустой ответ."""


class LLMImageUnsupportedError(LLMError):
    """Модель не поддерживает обработку изображений."""


class LLMAudioUnsupportedError(LLMError):
    """Провайдер/модель не поддерживает транскрибацию аудио."""


class AudioConversionError(Exception):
    """Ошибка конвертации аудио (ffmpeg)."""


class RagError(Exception):
    """Ошибка RAG-подсистемы (индексация, retrieval, генерация)."""
