"""Shared specialist-agent machinery. Each concrete agent module
(security_agent.py, quality_agent.py, ...) supplies only a system prompt
and an AgentType; this module owns the LLM call, structured-output
parsing, and defensive agent_type stamping that every specialist needs
identically. Adding a 5th specialist is "one new file with a prompt",
never a copy-paste of this plumbing.
"""
import logging
import threading
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import ValidationError

from pr_reviewer.domain.contracts import AgentType, Severity, Finding, SpecialistOutput
from pr_reviewer.orchestrator.state import ReviewState

logger = logging.getLogger(__name__)

# Keeps the diff well inside the model's context window. Large PRs need real
# chunking/retrieval later (the grounding problem) rather than a hard
# truncate, but truncating is the honest MVP behavior for now.
MAX_DIFF_CHARS = 40_000

# How many times to retry a single specialist call on a malformed/failed
# generation before giving up and degrading gracefully for that specialist
# only. Keeps one flaky call from taking down the whole review.
MAX_RETRIES = 2


def make_specialist_node(agent_type: AgentType, system_prompt: str, llm: ChatGroq):
    """Returns a LangGraph node function bound to one specialist's prompt
    and agent_type. All specialist modules call this - they never touch
    the LLM or the graph state directly.

    Uses JSON mode rather than tool-calling for structured output. Some
    Groq-hosted models (notably openai/gpt-oss-120b) use a "Harmony"
    response format with built-in tool channels, and on code-heavy diffs
    can hallucinate a tool call to a phantom tool named after the
    programming language in the diff (e.g. "csharp") instead of the
    SpecialistOutput schema tool. JSON mode has no tool namespace for the
    model to hallucinate into, so it sidesteps the bug entirely. Per Groq's
    JSON mode requirement, the word "json" must appear in the prompt.
    """
    structured_llm = llm.with_structured_output(SpecialistOutput, method="json_mode")

    def node(state: ReviewState) -> dict:
        diff = state["diff"][:MAX_DIFF_CHARS]
        messages = [
            SystemMessage(
                content=f"{system_prompt}\n\n"
                        f"Respond with a single JSON object matching this schema: "
                        f'{{"findings": [{{"agent_type": "{agent_type.value}", '
                        f'"severity": "critical|high|medium|low|info", "category": str, '
                        f'"summary": str, "file": str|null, "line": int|null, '
                        f'"confidence": float (0-1), "rationale": str, '
                        f'"suggestion": str|null}}]}}. '
                        f"Return {{\"findings\": []}} if there is nothing to report."
            ),
            HumanMessage(
                content=f"PR #{state['pr_number']}: {state['pr_title']}\n\n"
                        f"Diff (language of the changed files may vary):\n\n"
                        f"{diff}"
            ),
        ]

        last_error: Exception | None = None
        start = time.monotonic()
        logger.info(
            "specialist=%s status=start thread=%s",
            agent_type.value, threading.current_thread().name,
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result: SpecialistOutput = structured_llm.invoke(messages)
                # Stamp agent_type defensively in case the model omits/
                # mislabels it despite the schema.
                findings = [
                    f.model_copy(update={"agent_type": agent_type}) for f in result.findings
                ]
                logger.info(
                    "specialist=%s status=done thread=%s elapsed=%.2fs findings=%d",
                    agent_type.value, threading.current_thread().name,
                    time.monotonic() - start, len(findings),
                )
                return {"findings": findings}
            except (ValidationError, Exception) as exc:  # noqa: BLE001 - deliberately broad
                last_error = exc
                logger.warning(
                    "specialist=%s attempt=%d/%d failed: %s",
                    agent_type.value, attempt, MAX_RETRIES, exc,
                )

        # Every retry failed. Degrade gracefully: surface one LOW-confidence
        # INFO finding noting the specialist didn't run, rather than crashing
        # the whole graph and losing the other three specialists' findings.
        logger.error(
            "specialist=%s status=failed thread=%s elapsed=%.2fs attempts=%d: %s",
            agent_type.value, threading.current_thread().name,
            time.monotonic() - start, MAX_RETRIES, last_error,
        )
        return {
            "findings": [
                Finding(
                    agent_type=agent_type,
                    severity=Severity.INFO,
                    category="agent-error",
                    summary=f"The {agent_type.value} review could not complete.",
                    confidence=0.0,
                    rationale=f"LLM call failed after {MAX_RETRIES} attempts: {last_error}",
                )
            ]
        }

    return node