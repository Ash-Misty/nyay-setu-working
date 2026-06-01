"""
Layer 3: Parallel Research Engine
Sends all sub-questions to their assigned models simultaneously using asyncio.gather().
- Groq LPU for simple/factual questions
- Gemini for complex reasoning/precedent questions
"""

import asyncio
import logging
from google import genai
from groq import AsyncGroq
from config import (
    GROQ_API_KEY,
    GROQ_MODEL_FAST,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    RETRY_ENABLED,
    RETRY_MAX_ATTEMPTS,
    RETRY_DELAY_SECONDS,
    PROVIDER_ORDER,
)
from utils import CircuitBreaker, retry_transient, is_retryable_exception

logger = logging.getLogger(__name__)

# Initialize clients
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Module-level circuit-breakers for persistent state across requests
groq_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
gemini_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

LEGAL_SYSTEM_PROMPT = """You are an expert Indian legal advisor with deep knowledge of:
- Indian Penal Code (IPC) and Bharatiya Nyaya Sanhita (BNS) 2023
- Code of Criminal Procedure (CrPC) and Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023
- Code of Civil Procedure (CPC)
- Motor Vehicles Act (MVA)
- Indian Constitution including Fundamental Rights
- Consumer Protection Act
- Right to Information Act (RTI)

Provide precise, accurate, legally grounded answers. Quote specific section numbers where relevant.
Keep your answer focused, factual, and written for a common Indian citizen.
"""

KANOON_CONTEXT_PROMPT = """Use the Indian Kanoon context when it is relevant.
If a specific section or judgment is not present in the context, clearly say you cannot verify it.
"""


def _build_user_prompt(question: str, kanoon_context: str | None) -> str:
    if not kanoon_context:
        return question

    return (
        f"{question}\n\n"
        "INDIAN KANOON CONTEXT:\n"
        f"{kanoon_context}\n\n"
        "If the answer is not present in the context, say you cannot verify it."
    )


def _fallback_response(question: str, source: str) -> dict:
    """Structured fallback response when all retries are exhausted or circuit is open."""
    return {
        "question": question,
        "answer": "Our legal research service is temporarily unavailable. Please try again shortly.",
        "source": source,
        "error": "circuit_open",
        "is_fallback": True
    }


def build_provider_queue(primary_provider: str) -> list[str]:
    """Return ordered, deduplicated provider list with primary first."""
    ordered = [p for p in PROVIDER_ORDER if p in {"gemini", "groq"}]
    if primary_provider in ordered:
        ordered = [primary_provider] + [p for p in ordered if p != primary_provider]

    # If gemini is not configured, remove it
    if "gemini" in ordered and not gemini_client:
        ordered = [p for p in ordered if p != "gemini"]

    # Deduplicate while preserving order
    seen = set()
    result = []
    for p in ordered:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


async def _attempt_provider(provider: str, question: str, kanoon_context: str | None = None) -> dict:
    """Invoke a single provider once, honoring circuit-breaker state and returning a structured result or fallback."""
    if provider == "gemini":
        if not gemini_client:
            return _fallback_response(question, "gemini")
        if not gemini_breaker.is_available():
            logger.warning("[CircuitBreaker/Gemini] OPEN - skipping")
            return _fallback_response(question, "gemini")
        try:
            res = await _call_gemini_with_retry(question, kanoon_context)
            gemini_breaker.call_succeeded()
            return res
        except Exception as exc:
            gemini_breaker.call_failed()
            logger.error(f"[Research/Gemini] Provider attempt failed: {exc}")
            raise

    if provider == "groq":
        if not groq_breaker.is_available():
            logger.warning("[CircuitBreaker/Groq] OPEN - skipping")
            return _fallback_response(question, "groq")
        try:
            res = await _call_groq_with_retry(question, kanoon_context)
            groq_breaker.call_succeeded()
            return res
        except Exception as exc:
            groq_breaker.call_failed()
            logger.error(f"[Research/Groq] Provider attempt failed: {exc}")
            raise

    return _fallback_response(question, provider)


async def execute_with_fallback(question: str, kanoon_context: str | None = None, primary_provider: str = "gemini") -> dict:
    """Coordinator: try primary provider with retries, then fallback to secondary providers.

    Returns the first successful provider result or a unified fallback response when all fail.
    """
    providers = build_provider_queue(primary_provider)
    if not providers:
        return _fallback_response(question, "all_providers_failed")

    last_error = None
    for provider in providers:
        attempt = 1
        while attempt <= RETRY_MAX_ATTEMPTS:
            try:
                result = await _attempt_provider(provider, question, kanoon_context)
                # If provider returns a fallback-shaped response, treat as failure and try next provider
                if result.get("is_fallback"):
                    last_error = result.get("error", "fallback")
                    break
                return result
            except Exception as exc:
                last_error = exc
                # Decide whether to retry this exception
                if not RETRY_ENABLED or not is_retryable_exception(exc) or attempt >= RETRY_MAX_ATTEMPTS:
                    logger.error(f"[Research] Provider {provider} terminal error: {exc}")
                    break
                wait = RETRY_DELAY_SECONDS * attempt
                logger.warning(f"[Research] Transient error from {provider}, retry {attempt} in {wait}s: {exc}")
                await asyncio.sleep(wait)
                attempt += 1

        logger.warning(f"[Research] Provider {provider} exhausted, trying next provider if available")

    logger.error(f"[Research] All providers exhausted. Last error: {last_error}")
    return _fallback_response(question, "all_providers_failed")


@retry_transient
async def _call_groq_with_retry(question: str, kanoon_context: str | None = None) -> dict:
    """Helper to call Groq LPU with retry logic."""
    user_prompt = _build_user_prompt(question, kanoon_context)
    response = await groq_client.chat.completions.create(
        model=GROQ_MODEL_FAST,
        messages=[
            {"role": "system", "content": f"{LEGAL_SYSTEM_PROMPT}\n\n{KANOON_CONTEXT_PROMPT}"},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=800
    )
    return {
        "question": question,
        "answer": response.choices[0].message.content.strip(),
        "source": "groq",
        "error": None
    }


@retry_transient
async def _call_gemini_with_retry(question: str, kanoon_context: str | None = None) -> dict:
    """Helper to call Gemini with retry logic."""
    user_prompt = _build_user_prompt(question, kanoon_context)
    full_prompt = (
        f"{LEGAL_SYSTEM_PROMPT}\n\n"
        f"{KANOON_CONTEXT_PROMPT}\n\n"
        f"Question: {user_prompt}"
    )
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 800
            }
        )
    )
    return {
        "question": question,
        "answer": response.text.strip(),
        "source": "gemini",
        "error": None
    }


async def call_groq_async(question: str, kanoon_context: str | None = None) -> dict:
    """Call Groq LPU with circuit breaker + retry logic."""
    if not groq_breaker.is_available():
        logger.warning("[CircuitBreaker/Groq] OPEN - fast failing")
        return _fallback_response(question, "groq")
    try:
        result = await _call_groq_with_retry(question, kanoon_context)
        groq_breaker.call_succeeded()
        return result
    except Exception as e:
        groq_breaker.call_failed()
        logger.error(f"[Research/Groq] Failed after retries: {e}")
        return _fallback_response(question, "groq")


async def call_gemini_async(question: str, kanoon_context: str | None = None) -> dict:
    """Call Gemini with circuit breaker + retry logic, falls back to Groq."""
    if not gemini_client:
        return await call_groq_async(question, kanoon_context)
    if not gemini_breaker.is_available():
        logger.warning("[CircuitBreaker/Gemini] OPEN — falling back to Groq")
        return await call_groq_async(question, kanoon_context)
    try:
        result = await _call_gemini_with_retry(question, kanoon_context)
        gemini_breaker.call_succeeded()
        return result
    except Exception as e:
        gemini_breaker.call_failed()
        logger.error(f"[Research/Gemini] Failed after retries: {e}")
        return await call_groq_async(question, kanoon_context)


async def run_parallel_research(
    routed_questions: list[dict],
    kanoon_context: str | None = None
) -> list[dict]:
    """
    Run all sub-questions in parallel using asyncio.gather().
    Returns results in the same order as input.
    """
    tasks = []
    for item in routed_questions:
        question = item["question"]
        model = item["model"]

        # Use unified coordinator to handle retries and fallback ordering
        tasks.append(execute_with_fallback(question, kanoon_context, primary_provider=model))

    results = await asyncio.gather(*tasks, return_exceptions=False)
    return list(results)