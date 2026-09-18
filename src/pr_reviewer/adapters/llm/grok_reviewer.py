"""Concrete Reviewer that runs a (currently single-node) LangGraph graph,
calling a Groq-hosted open model through LangChain's ChatGroq.

Why a graph for one node? Because this is the seam Phase 8 grows from: to
add the security/quality/tests/docs specialist split later, you add more
nodes here and fan them out with LangGraph's Send API before the aggregator
node - nothing outside this file has to change, since GroqReviewer still
satisfies the same Reviewer port.
"""
import os
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from pr_reviewer.domain.interfaces import Reviewer
from pr_reviewer.domain.models import PullRequestEvent, Review
from dotenv import load_dotenv
load_dotenv()
# Keeps the diff well inside the model's context window. Large PRs need real
# chunking/retrieval later (the grounding problem - Phase 6) rather than a
# hard truncate, but truncating is the honest MVP behavior for now.
MAX_DIFF_CHARS = 40_000

_SYSTEM_PROMPT = """You are a careful senior software engineer reviewing a pull request diff.
Point out real bugs, security issues, and missed edge cases - skip nitpicks and style
comments a linter would already catch. If the diff looks fine, say so briefly.
Keep the review under 200 words, in markdown, suitable for posting as a GitHub comment."""


class ReviewState(TypedDict):
    """The typed state that flows through the graph. Small and flat on
    purpose - this is what LangGraph would checkpoint to Redis once you
    wire that up (see 3.2 in the design doc)."""
    pr_number: int
    pr_title: str
    diff: str
    summary: str


def _build_graph(llm: ChatGroq):
    def review_node(state: ReviewState) -> dict:
        diff = state["diff"][:MAX_DIFF_CHARS]
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=f"PR #{state['pr_number']}: {state['pr_title']}\n\n```diff\n{diff}\n```"
            ),
        ]
        response = llm.invoke(messages)
        return {"summary": response.content}

    graph = StateGraph(ReviewState)
    graph.add_node("review", review_node)
    graph.set_entry_point("review")
    graph.add_edge("review", END)
    # Phase 8 sketch:
    #   graph.add_node("security", security_node); graph.add_node("quality", quality_node); ...
    #   graph.add_conditional_edges(START, lambda s: [Send("security", s), Send("quality", s), ...])
    #   graph.add_node("aggregate", aggregate_node)
    #   graph.add_edge(["security", "quality", "tests", "docs"], "aggregate")
    return graph.compile()


class GroqReviewer(Reviewer):
    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-oss-120b"):
        # llama-3.3-70b-versatile is Groq's general-purpose recommendation.
        # For heavier reasoning, openai/gpt-oss-120b is also on the free tier.
        llm = ChatGroq(
            model=model or os.getenv("GROK_MODEL"),
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
        self._graph = _build_graph(llm)

    def review(self, event: PullRequestEvent, diff: str) -> Review:
        result = self._graph.invoke({
            "pr_number": event.pr_number,
            "pr_title": event.pr_title,
            "diff": diff,
            "summary": "",
        })
        return Review(pr_number=event.pr_number, summary=result["summary"], verdict="comment")