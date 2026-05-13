"""Единое форматирование пользовательских сообщений (ответ/баланс/список)."""
from __future__ import annotations

from datetime import time as dt_time
from typing import Sequence

from models import Transaction, TransactionResponse, TransactionType

TELEGRAM_MESSAGE_LIMIT = 4000


def _balance_of(transactions: Sequence[Transaction]) -> float:
    return sum(
        t.amount if t.type is TransactionType.INCOME else -t.amount
        for t in transactions
    )


def _format_amount(value: float) -> str:
    return f"{value:.0f}" if value == int(value) else f"{value:.2f}"


def _pluralize_found(count: int) -> str:
    suffix = "и" if count > 1 else ""
    return f"✅ Найдено и сохранено {count} транзакция{suffix}"


class ReportFormatter:
    """Чистые функции форматирования без I/O."""

    def format_answer(
        self,
        response: TransactionResponse,
        user_transactions: Sequence[Transaction],
    ) -> str:
        """«ответ + статус + баланс» — единый блок для text/image/voice хендлеров."""
        status = (
            _pluralize_found(len(response.transactions))
            if response.transactions
            else "ℹ️ Транзакции не найдены"
        )
        balance = _balance_of(user_transactions)
        return (
            f"{response.answer}\n\n{status}\n💵 Баланс: {_format_amount(balance)} руб."
        )

    def format_balance_report(self, transactions: Sequence[Transaction]) -> str:
        if not transactions:
            return (
                "💵 У вас пока нет транзакций.\n\n"
                "Отправьте сообщение с транзакцией или изображение чека для начала учета."
            )

        total_income = sum(
            t.amount for t in transactions if t.type is TransactionType.INCOME
        )
        total_expense = sum(
            t.amount for t in transactions if t.type is TransactionType.EXPENSE
        )
        balance = total_income - total_expense

        category_stats: dict[str, float] = {}
        for t in transactions:
            delta = t.amount if t.type is TransactionType.INCOME else -t.amount
            category_stats[t.category] = category_stats.get(t.category, 0.0) + delta

        lines = [
            "💵 **Отчет о балансе**\n",
            f"📊 Баланс: {balance:.2f} руб.",
            f"💰 Доходы: {total_income:.2f} руб.",
            f"💸 Расходы: {total_expense:.2f} руб.",
            f"\n📈 Всего транзакций: {len(transactions)}",
            "\n**Статистика по категориям:**",
        ]
        sorted_categories = sorted(
            category_stats.items(), key=lambda x: abs(x[1]), reverse=True
        )
        for category, amount in sorted_categories:
            sign = "💰" if amount > 0 else "💸"
            lines.append(f"{sign} {category}: {amount:+.2f} руб.")
        return "\n".join(lines)

    def format_transaction_list(
        self, transactions: Sequence[Transaction]
    ) -> list[str]:
        """Возвращает список готовых к отправке сообщений (учитывает лимит Telegram)."""
        if not transactions:
            return [
                "📋 У вас пока нет транзакций.\n\n"
                "Отправьте сообщение с транзакцией или изображение чека для начала учета."
            ]

        sorted_tx = sorted(
            transactions,
            key=lambda t: (t.date, t.time or dt_time(0, 0)),
            reverse=True,
        )
        header = f"📋 **Все транзакции** ({len(transactions)} шт.)\n"
        entries = [header] + [self._format_entry(i, t) for i, t in enumerate(sorted_tx, 1)]
        return self._split_into_messages(entries)

    @staticmethod
    def _format_entry(index: int, t: Transaction) -> str:
        date_str = t.date.strftime("%d.%m.%Y")
        time_str = f" {t.time.strftime('%H:%M')}" if t.time else ""
        sign = "💰" if t.type is TransactionType.INCOME else "💸"
        type_str = "Доход" if t.type is TransactionType.INCOME else "Расход"
        amount_str = f"{t.amount:.2f}".rstrip("0").rstrip(".")
        desc_str = f"\n   {t.description}" if t.description else ""
        return (
            f"{index}. {sign} **{type_str}** {amount_str} руб.\n"
            f"   📅 {date_str}{time_str}\n"
            f"   🏷️ {t.category}{desc_str}"
        )

    @staticmethod
    def _split_into_messages(entries: list[str]) -> list[str]:
        joined = "\n\n".join(entries)
        if len(joined) <= TELEGRAM_MESSAGE_LIMIT:
            return [joined]

        parts: list[str] = []
        current: list[str] = [entries[0]]
        current_len = len(entries[0])
        for entry in entries[1:]:
            extra = len(entry) + 2  # "\n\n"
            if current_len + extra > TELEGRAM_MESSAGE_LIMIT:
                parts.append("\n\n".join(current))
                current = [entry]
                current_len = len(entry)
                continue
            current.append(entry)
            current_len += extra
        if current:
            parts.append("\n\n".join(current))
        return parts
