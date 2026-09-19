"""Tests specialist. Asks 'what's untested?' - coverage gaps introduced by
this specific diff, not a general audit of the test suite.
"""
from langchain_groq import ChatGroq

from pr_reviewer.agents.base_agent import make_specialist_node
from pr_reviewer.domain.contracts import AgentType

SYSTEM_PROMPT = (
    "You are a senior engineer reviewing a pull request diff for test coverage. "
    "Look only for missing test cases, untested edge conditions, brittle "
    "assertions, and coverage gaps introduced by this diff. If existing tests "
    "adequately cover the change, return an empty findings list."
)


def build_node(llm: ChatGroq):
    return make_specialist_node(AgentType.TESTS, SYSTEM_PROMPT, llm)
