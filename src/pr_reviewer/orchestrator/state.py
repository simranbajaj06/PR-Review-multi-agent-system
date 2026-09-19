"""The typed state that flows through the multi-agent graph. Small and
flat on purpose - this is what LangGraph would checkpoint to Redis once
you wire that up for crash recovery.
"""
from typing import Annotated, TypedDict

from pr_reviewer.domain.contracts import Finding


def _merge_findings(a: list[Finding], b: list[Finding]) -> list[Finding]:
    """Reducer LangGraph applies when multiple parallel branches write to
    `findings` in the same step. Each of the four specialist nodes returns
    only its own findings; LangGraph concatenates them via this function
    rather than the branches racing to overwrite one another.
    """
    return a + b


class ReviewState(TypedDict):
    pr_number: int
    pr_title: str
    diff: str
    findings: Annotated[list[Finding], _merge_findings]
    overall_confidence: float
    needs_human_review: bool
    summary: str
