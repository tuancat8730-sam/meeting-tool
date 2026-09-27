"""LiteLLM wrapper: structured (Pydantic) output, transient-error retry, cost accounting."""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import litellm
import stamina
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.Timeout,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.ServiceUnavailableError,
)
REPAIR_PROMPT = (
    "Kết quả trên không đúng định dạng JSON yêu cầu. Lỗi:\n{errors}\n"
    "Hãy trả lại TOÀN BỘ kết quả dưới dạng JSON hợp lệ đúng schema, không kèm giải thích."
)

CompletionFn = Callable[..., Any]


class LLMOutputError(RuntimeError):
    pass


@dataclass(frozen=True)
class StructuredResult[T: BaseModel]:
    value: T
    cost_usd: float | None
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class _Call:
    content: str
    cost_usd: float | None
    prompt_tokens: int
    completion_tokens: int


def extract_json(raw: str) -> str:
    """Pull the JSON object out of fenced or chatty model output."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    first, last = raw.find("{"), raw.rfind("}")
    return raw[first : last + 1] if 0 <= first < last else raw


def _sum_costs(*costs: float | None) -> float | None:
    return None if any(c is None for c in costs) else sum(c for c in costs if c is not None)


class LLMClient:
    def __init__(
        self,
        model: str,
        timeout_s: float = 600.0,
        completion_fn: CompletionFn | None = None,
        max_attempts: int = 5,
    ) -> None:
        self.model = model
        self._timeout_s = timeout_s
        self._completion = completion_fn or litellm.completion
        self._max_attempts = max_attempts

    def complete_structured[T: BaseModel](
        self, messages: list[dict[str, Any]], schema: type[T]
    ) -> StructuredResult[T]:
        """Call the model and validate its output against `schema`, repairing once if invalid."""
        first = self._call(messages, schema)
        try:
            return self._result(schema.model_validate_json(extract_json(first.content)), first)
        except ValidationError as exc:
            logger.warning("%s returned invalid output, asking it to repair", self.model)
            repair_messages = [
                *messages,
                {"role": "assistant", "content": first.content},
                {"role": "user", "content": REPAIR_PROMPT.format(errors=exc)},
            ]
        second = self._call(repair_messages, schema)
        try:
            value = schema.model_validate_json(extract_json(second.content))
        except ValidationError as exc:
            raise LLMOutputError(
                f"{self.model} returned output that does not match {schema.__name__} "
                f"after one repair attempt: {exc.error_count()} validation errors"
            ) from exc
        return StructuredResult(
            value=value,
            cost_usd=_sum_costs(first.cost_usd, second.cost_usd),
            prompt_tokens=first.prompt_tokens + second.prompt_tokens,
            completion_tokens=first.completion_tokens + second.completion_tokens,
        )

    @staticmethod
    def _result[T: BaseModel](value: T, call: _Call) -> StructuredResult[T]:
        return StructuredResult(
            value=value,
            cost_usd=call.cost_usd,
            prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
        )

    def _response_format(self, schema: type[BaseModel]) -> Any:
        try:
            supported = litellm.supports_response_schema(model=self.model)
        except Exception:
            supported = False
        return schema if supported else {"type": "json_object"}

    def _call(self, messages: list[dict[str, Any]], schema: type[BaseModel]) -> _Call:
        response_format = self._response_format(schema)
        for attempt in stamina.retry_context(on=TRANSIENT_ERRORS, attempts=self._max_attempts):
            with attempt:
                response = self._completion(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    temperature=0,
                    timeout=self._timeout_s,
                )
        usage = getattr(response, "usage", None)
        return _Call(
            content=response.choices[0].message.content or "",
            cost_usd=self._cost(response),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def _cost(self, response: Any) -> float | None:
        try:
            return float(litellm.completion_cost(completion_response=response))
        except Exception as exc:  # pricing table may not know new/custom models
            logger.warning("Cost unknown for %s: %s", self.model, exc)
            return None
