from datetime import date as date_type, time as time_type
from enum import Enum

from pydantic import BaseModel, Field


class TransactionType(str, Enum):
    INCOME = "income"      # доход
    EXPENSE = "expense"    # расход


class TransactionFrequency(str, Enum):
    DAILY = "daily"           # повседневные
    PERIODIC = "periodic"     # периодические
    ONE_TIME = "one_time"     # разовые


class Transaction(BaseModel):
    date: date_type                      # дата транзакции
    time: time_type | None = None        # время (опционально)
    type: TransactionType                # доход/расход
    amount: float = Field(gt=0)          # сумма (строго положительная)
    frequency: TransactionFrequency      # тип (повседневные, периодические, разовые)
    category: str                        # категория (продукты, рестораны, такси и т.д.)
    description: str = ""                # описание транзакции

class TransactionResponse(BaseModel):
    transactions: list[Transaction]  # список транзакций (всегда должен быть, пустой [] если не найдено)
    answer: str                     # текстовый ответ пользователю (обязателен)

