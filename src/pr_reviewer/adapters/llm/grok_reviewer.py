"""Concrete Reviewer adapter: builds the Groq LLM client and the
multi-agent orchestrator graph, and translates between the domain's
PullRequestEvent/Review and the graph's internal ReviewState. All review
*logic* lives in agents/ and orchestrator/ - this file is just wiring
plus the Reviewer port implementation.
"""
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from pr_reviewer.domain.interfaces import Reviewer
from pr_reviewer.domain.models import PullRequestEvent, Review
from pr_reviewer.orchestrator.graph import build_graph

load_dotenv()


class GroqReviewer(Reviewer):
    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-oss-120b"):
        # llama-3.3-70b-versatile is Groq's general-purpose recommendation.
        # For heavier reasoning, openai/gpt-oss-120b is also on the free tier.
        llm = ChatGroq(
            model=model or os.getenv("GROK_MODEL"),
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
        self._graph = build_graph(llm)

    def review(self, event: PullRequestEvent, diff: str) -> Review:
        result = self._graph.invoke({
            "pr_number": event.pr_number,
            "pr_title": event.pr_title,
            "diff": diff,
            "findings": [],
            "overall_confidence": 1.0,
            "needs_human_review": False,
            "summary": "",
        })
        # NOTE: adjust to your actual Review model. If it doesn't yet have a
        # field for the gate, either add `needs_human_review: bool` to it,
        # or fold the decision into `verdict` as done here (e.g. "comment"
        # vs "hold") - whichever your GitHub-posting step already expects.
        verdict = "hold" if result["needs_human_review"] else "comment"
        return Review(pr_number=event.pr_number, summary=result["summary"], verdict=verdict)
