"""RAGAS evaluation через Langfuse dataset.run_experiment().

Берёт датасет из Langfuse, прогоняет каждый item через RAG,
оценивает RAGAS-метриками и сохраняет scores привязанными к dataset run.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langfuse import Evaluation, Langfuse
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness, AnswerRelevancy, AnswerSimilarity, AnswerCorrectness,
LLMContextPrecisionWithoutReference, LLMContextRecall
)


from ragas.metrics.base import MetricWithEmbeddings, MetricWithLLM, SingleTurnMetric


from ragas.run_config import RunConfig

from config import Settings
from dataset_item import ANSWER_KEY, QUESTION_KEY
from rag.context_retriever import ContextRetriever
from rag.embeddings_factory import create_embeddings
from rag_service import RagService

LANGFUSE_DATASET_NAME = "05-rag-qa-dataset"
EVAL_CHAT_ID = -1

logger = logging.getLogger(__name__)


class RagEvaluator:
    """Прогоняет QA-датасет из Langfuse через RAG и оценивает RAGAS-метриками."""

    def __init__(self, settings: Settings, rag_service: RagService) -> None:
        self._settings = settings
        self._rag_service = rag_service

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, float]:
        """Полный цикл: Langfuse dataset → RAG → RAGAS → scores в Langfuse."""
        langfuse = Langfuse()
        dataset = langfuse.get_dataset(LANGFUSE_DATASET_NAME)
        if not dataset.items:
            raise ValueError(
                f"Датасет '{LANGFUSE_DATASET_NAME}' пуст в Langfuse. "
                "Сначала выполните make dataset && make upload."
            )

        metrics = self._init_metrics()

        logger.info(
            "Запускаю evaluation: dataset=%s, items=%d, metrics=%d",
            LANGFUSE_DATASET_NAME, len(dataset.items), len(metrics),
        )

        result = dataset.run_experiment(
            name="ragas-evaluation",
            task=self._make_task(),
            evaluators=self._make_evaluators(metrics),
            max_concurrency=1,  # RAG-сервис не потокобезопасен
        )

        avg_scores = self._compute_averages(result)
        logger.info("RAGAS evaluation done: %s", avg_scores)
        return avg_scores

    @staticmethod
    def format_results(avg_scores: dict[str, float]) -> str:
        """Форматирует средние метрики для отображения в чате."""
        lines = ["📊 RAGAS Evaluation Results\n"]
        for name, value in sorted(avg_scores.items()):
            indicator = "🟢" if value >= 0.8 else "🟡" if value >= 0.6 else "🔴"
            lines.append(f"{indicator} {name}: {value:.4f}")
        lines.append("\n📖 Легенда:")
        lines.append("• Faithfulness — ответ опирается на документы")
        lines.append("• AnswerRelevancy — ответ релевантен вопросу")
        lines.append("• AnswerCorrectness — совпадение с эталоном")
        lines.append("• AnswerSimilarity — семантическая близость к эталону")
        lines.append("• ContextRecall — retriever нашёл все нужные фрагменты")
        lines.append("• ContextPrecision — в найденном нет лишнего шума")
        lines.append("\n🟢 0.8+ отлично · 🟡 0.6–0.8 хорошо · 🔴 <0.6 требует улучшений")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # private
    # ------------------------------------------------------------------

    def _init_metrics(self) -> list[SingleTurnMetric]:
        """Создаёт и инициализирует RAGAS-метрики с LLM/embeddings."""
        ragas_llm = LangchainLLMWrapper(ChatOpenAI(
            model=self._settings.ragas_llm_model,
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url,
            temperature=0,
        ))
        ragas_embeddings = LangchainEmbeddingsWrapper(create_embeddings(
            self._settings.ragas_embeddings_provider,
            self._settings.ragas_embeddings_model,
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url,
        ))

        metrics: list[SingleTurnMetric] = [
            Faithfulness(),
            AnswerRelevancy(),
            AnswerCorrectness(),
            AnswerSimilarity(),
            LLMContextRecall(),
            LLMContextPrecisionWithoutReference(),
        ]

        run_config = RunConfig()
        for metric in metrics:
            if isinstance(metric, MetricWithLLM) and metric.llm is None:
                metric.llm = ragas_llm
            if isinstance(metric, MetricWithEmbeddings) and metric.embeddings is None:
                metric.embeddings = ragas_embeddings
            metric.init(run_config)

        return metrics

    def _make_task(self):
        """Возвращает task-функцию для run_experiment: item → RAG-ответ."""
        rag_service = self._rag_service

        def task(*, item, **_kwargs) -> dict[str, Any]:
            question = item.input.get(QUESTION_KEY, "")
            rag_service.reset(EVAL_CHAT_ID)
            response, chunks = rag_service.answer(EVAL_CHAT_ID, question)
            contexts = [ContextRetriever.format([c]) for c in chunks]
            return {"response": response, "contexts": contexts}

        return task

    def _make_evaluators(self, metrics: list[SingleTurnMetric]) -> list:
        """Создаёт Langfuse evaluator-функции из RAGAS-метрик."""
        evaluators = []
        for metric in metrics:
            evaluators.append(_ragas_evaluator(metric))
        return evaluators

    @staticmethod
    def _compute_averages(result) -> dict[str, float]:
        """Считает средние значения из ExperimentResult."""
        totals: dict[str, list[float]] = {}
        for item_result in result.item_results:
            for evaluation in item_result.evaluations:
                if evaluation.value is not None:
                    totals.setdefault(evaluation.name, []).append(
                        float(evaluation.value)
                    )
        return {
            name: sum(values) / len(values)
            for name, values in totals.items()
            if values
        }


def _ragas_evaluator(metric: SingleTurnMetric):
    """Фабрика: RAGAS-метрика → Langfuse evaluator-функция."""

    def evaluator(
        *, input, output, expected_output=None, metadata=None, **_kwargs
    ) -> Evaluation:
        question = input.get(QUESTION_KEY, "") if isinstance(input, dict) else str(input)
        reference = (
            expected_output.get(ANSWER_KEY, "")
            if isinstance(expected_output, dict)
            else str(expected_output or "")
        )
        response = output.get("response", "") if isinstance(output, dict) else str(output)
        contexts = output.get("contexts", []) if isinstance(output, dict) else []

        sample = SingleTurnSample(
            user_input=question,
            response=response,
            retrieved_contexts=contexts,
            reference=reference,
        )

        try:
            score = metric.single_turn_score(sample)
        except Exception:
            logger.exception("RAGAS metric '%s' failed", metric.name)
            return Evaluation(name=metric.name, value=0.0, comment="metric error")

        return Evaluation(
            name=metric.name,
            value=score,
            comment=f"RAGAS {metric.name}",
        )

    evaluator.__name__ = f"ragas_{metric.name}"
    return evaluator
