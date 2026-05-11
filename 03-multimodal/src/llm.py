import logging
from openai import AsyncOpenAI
from openai import APIError, InternalServerError
from config import config
from models import TransactionResponse

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL
)

async def get_transaction_response_text(
    last_message: str,
    message_history: list[dict]
) -> TransactionResponse:
    try:
        response = await client.chat.completions.create(
            model=config.MODEL_TEXT,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT_TEXT},
                *message_history[-10:],  # последние 10 сообщений для контекста
                {"role": "user", "content": last_message}
            ],
            response_format={"type": "json_schema", "json_schema": {
                "name": "transaction_response",
                "schema": TransactionResponse.model_json_schema(),
                "strict": True
            }}
        )
        raw_content = response.choices[0].message.content
        logger.info(f"Raw LLM response (length: {len(raw_content) if raw_content else 0}): {raw_content[:1000] if raw_content else 'EMPTY'}")
        
        # Проверяем что ответ не пустой
        if not raw_content or not raw_content.strip():
            logger.error("LLM returned empty response")
            raise ValueError("LLM returned empty response")
        
        try:
            # Парсим JSON ответ
            import json
            parsed_json = json.loads(raw_content)
            
            # Обрабатываем случай, когда поле transactions отсутствует
            if "transactions" not in parsed_json:
                logger.warning("Field 'transactions' missing in LLM response, adding empty list")
                parsed_json["transactions"] = []
            
            # Убеждаемся, что answer есть
            if "answer" not in parsed_json:
                logger.warning("Field 'answer' missing in LLM response, adding default")
                parsed_json["answer"] = "Обработал ваше сообщение."
            
            parsed_response = TransactionResponse.model_validate(parsed_json)
            logger.info(f"Successfully parsed TransactionResponse: transactions={len(parsed_response.transactions)}")
            return parsed_response
        except json.JSONDecodeError as json_error:
            # Детальное логирование проблемы с JSON
            logger.error(f"Failed to parse JSON from LLM response: {json_error}")
            logger.error(f"Full response content ({len(raw_content)} chars): {raw_content}")
            logger.error(f"First 200 chars: {raw_content[:200]}")
            logger.error(f"Last 200 chars: {raw_content[-200:]}")
            raise
        except Exception as parse_error:
            # Детальное логирование для других ошибок парсинга
            logger.error(f"Failed to parse LLM response as TransactionResponse: {parse_error}")
            logger.error(f"Full response content ({len(raw_content)} chars): {raw_content}")
            logger.error(f"First 200 chars: {raw_content[:200]}")
            logger.error(f"Last 200 chars: {raw_content[-200:]}")
            raise
    except (APIError, InternalServerError) as e:
        logger.error(f"LLM API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error calling LLM: {e}", exc_info=True)
        raise

async def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "voice.ogg"
) -> str:
    """Транскрибация голосового сообщения через RouterAI chat completions API."""
    try:
        import base64
        import subprocess
        import tempfile
        import os
        
        # Определяем формат из расширения файла
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'ogg'
        
        logger.info(
            f"Transcribing audio: filename={filename}, size={len(audio_bytes)} bytes, "
            f"model={config.MODEL_AUDIO}, source_format={ext}"
        )
        
        # Конвертируем в WAV если формат не поддерживается напрямую
        # RouterAI поддерживает только 'wav' и 'mp3'
        if ext in ('ogg', 'flac', 'm4a'):
            logger.info(f"Converting {ext} to WAV for RouterAI compatibility")
            with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as src_file:
                src_file.write(audio_bytes)
                src_path = src_file.name
            
            wav_path = src_path.replace('.ogg', '.wav')
            try:
                # Конвертируем через ffmpeg в WAV формат
                result = subprocess.run(
                    [
                        'ffmpeg', '-y', '-i', src_path,
                        '-acodec', 'pcm_s16le',  # PCM 16-bit
                        '-ar', '16000',           # 16kHz sample rate (оптимально для распознавания)
                        '-ac', '1',               # mono
                        wav_path
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg conversion failed: {result.stderr}")
                    raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
                
                # Читаем сконвертированный файл
                with open(wav_path, 'rb') as f:
                    audio_bytes = f.read()
                
                audio_format = 'wav'
                logger.info(f"Successfully converted to WAV, new size: {len(audio_bytes)} bytes")
            finally:
                # Чистим временные файлы
                for path in [src_path, wav_path]:
                    if os.path.exists(path):
                        os.unlink(path)
        else:
            # Уже в поддерживаемом формате
            audio_format = ext if ext in ('wav', 'mp3') else 'wav'
        
        # Кодируем аудио в base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Используем chat completions API с аудио в content
        response = await client.chat.completions.create(
            model=config.MODEL_AUDIO,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Расшифруй этот аудиофайл. Верни только текст расшифровки без дополнительных комментариев."},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_base64,
                                "format": audio_format
                            }
                        }
                    ]
                }
            ]
        )
        
        text = response.choices[0].message.content or ""
        logger.info(f"Transcription result (length {len(text)}): {text[:500]}")
        return text
    except (APIError, InternalServerError) as e:
        logger.error(f"LLM API error during transcription: {e}")
        raise
    except Exception as e:
        logger.error(f"Error calling transcription API: {e}", exc_info=True)
        raise


async def get_transaction_response_image(
    image_base64: str,
    message_history: list[dict]
) -> TransactionResponse:
    try:
        schema = TransactionResponse.model_json_schema()
        logger.info(f"Using model: {config.MODEL_IMAGE}, base_url: {config.OPENAI_BASE_URL}")
        
        # Логируем размер изображения в более понятном формате
        image_size_bytes = len(image_base64.encode('utf-8')) * 3 // 4  # примерная оценка
        image_size_kb = image_size_bytes / 1024
        logger.info(f"Image size: ~{image_size_kb:.1f} KB ({len(image_base64)} base64 chars)")
        logger.info(f"Message history length: {len(message_history)} messages")
        
        response = await client.chat.completions.create(
            model=config.MODEL_IMAGE,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT_IMAGE},
                *message_history[-10:],  # последние 10 сообщений для контекста
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
                "schema": schema,
                "strict": True  # Используем strict для лучшего соответствия схеме
            }}
        )
        
        # Логируем информацию о response объекте
        logger.info(f"Response object: {response}")
        logger.info(f"Response choices count: {len(response.choices)}")
        if response.choices:
            logger.info(f"First choice finish_reason: {response.choices[0].finish_reason}")
            logger.info(f"First choice message role: {response.choices[0].message.role}")
        
        raw_content = response.choices[0].message.content
        logger.info(f"Raw LLM response for image (length: {len(raw_content) if raw_content else 0}): {raw_content[:1000] if raw_content else 'EMPTY'}")
        
        # Проверяем что ответ не пустой
        if not raw_content or not raw_content.strip():
            logger.error("LLM returned empty response for image")
            logger.error(f"Response object details: {response}")
            logger.error(f"Finish reason: {response.choices[0].finish_reason if response.choices else 'no choices'}")
            raise ValueError("LLM returned empty response")
        
        try:
            # Парсим JSON ответ
            import json
            parsed_json = json.loads(raw_content)
            
            # Обрабатываем случай, когда поле transactions отсутствует
            if "transactions" not in parsed_json:
                logger.warning("Field 'transactions' missing in LLM response, adding empty list")
                parsed_json["transactions"] = []
            
            # Убеждаемся, что answer есть
            if "answer" not in parsed_json:
                logger.warning("Field 'answer' missing in LLM response, adding default")
                parsed_json["answer"] = "Обработал изображение."
            
            parsed_response = TransactionResponse.model_validate(parsed_json)
            logger.info(f"Successfully parsed TransactionResponse for image: transactions={len(parsed_response.transactions)}")
            return parsed_response
        except json.JSONDecodeError as json_error:
            # Детальное логирование проблемы с JSON
            logger.error(f"Failed to parse JSON from LLM response for image: {json_error}")
            logger.error(f"Full response content ({len(raw_content)} chars): {raw_content}")
            logger.error(f"First 200 chars: {raw_content[:200]}")
            logger.error(f"Last 200 chars: {raw_content[-200:]}")
            raise
        except Exception as parse_error:
            # Детальное логирование для других ошибок парсинга
            logger.error(f"Failed to parse LLM response as TransactionResponse for image: {parse_error}")
            logger.error(f"Full response content ({len(raw_content)} chars): {raw_content}")
            logger.error(f"First 200 chars: {raw_content[:200]}")
            logger.error(f"Last 200 chars: {raw_content[-200:]}")
            raise
    except (APIError, InternalServerError) as e:
        logger.error(f"LLM API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error calling LLM: {e}", exc_info=True)
        raise

