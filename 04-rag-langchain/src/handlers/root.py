"""Корневой роутер: композиция финансового и RAG-подроутеров."""
from __future__ import annotations

from aiogram import Router

from finance_service import FinanceService
from handlers.finance_router import build_finance_router
from handlers.rag_router import build_rag_router
from rag_service import RagService
from report_formatter import ReportFormatter


def build_router(
    service: FinanceService,
    formatter: ReportFormatter,
    rag_service: RagService,
) -> Router:
    router = Router()
    # RAG-роутер первым: его команды не должны перехватываться catch-all
    # финансового роутера (@router.message()).
    router.include_router(build_rag_router(rag_service))
    router.include_router(build_finance_router(service, formatter))
    return router
