"""Merges findings from all specialist agents into one review: dedupes
overlapping file/line findings (keeping the highest-confidence one),
computes an overall confidence, and applies the confidence/severity gate
that decides whether a review can auto-post or must be held for human
approval.
"""
from pr_reviewer.domain.contracts import Finding, Severity
from pr_reviewer.orchestrator.state import ReviewState

# Below this overall confidence, or on any CRITICAL finding, the review is
# held for human approval instead of auto-posted. Tune once you have real
# data on how often the model is right vs. overconfident.
CONFIDENCE_THRESHOLD = 0.6

_SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}


def _dedupe(findings: list[Finding]) -> list[Finding]:
    best: dict[tuple[str | None, int | None], Finding] = {}
    for f in findings:
        key = (f.file, f.line)
        if key not in best or f.confidence > best[key].confidence:
            best[key] = f
    return sorted(best.values(), key=lambda f: _SEVERITY_ORDER[f.severity])


def _render_markdown(findings: list[Finding], needs_human_review: bool) -> str:
    lines = []
    if needs_human_review:
        lines.append(
            "> ⚠️ This review contains a critical or low-confidence finding "
            "and is held for human approval before posting.\n"
        )
    for f in findings:
        location = f" (`{f.file}:{f.line}`)" if f.file else ""
        lines.append(f"**[{f.severity.value.upper()}] {f.category}**{location}")
        lines.append(f.summary)
        lines.append(f"_{f.rationale}_")
        if f.suggestion:
            lines.append(f"Suggested fix: {f.suggestion}")
        lines.append(f"<sub>{f.agent_type.value} · confidence {f.confidence:.2f}</sub>\n")
    return "\n".join(lines)


def aggregate_node(state: ReviewState) -> dict:
    findings = _dedupe(state["findings"])

    if not findings:
        return {
            "findings": [],
            "overall_confidence": 1.0,
            "needs_human_review": False,
            "summary": "No issues found. The diff looks fine.",
        }

    overall_confidence = min(f.confidence for f in findings)
    has_critical = any(f.severity == Severity.CRITICAL for f in findings)
    needs_human_review = has_critical or overall_confidence < CONFIDENCE_THRESHOLD

    return {
        "findings": findings,
        "overall_confidence": overall_confidence,
        "needs_human_review": needs_human_review,
        "summary": _render_markdown(findings, needs_human_review),
    }
