"""Docs specialist. Asks 'will the next reader understand?' - missing or
stale documentation introduced by this diff.
"""
from langchain_groq import ChatGroq

from pr_reviewer.agents.base_agent import make_specialist_node
from pr_reviewer.domain.contracts import AgentType

SYSTEM_PROMPT = (
    "You are a senior engineer reviewing a pull request diff for documentation. "
    "Look only for missing docstrings on new public functions/classes, outdated "
    "comments that no longer match the code, and undocumented public API changes. "
    "If documentation is adequate, return an empty findings list."
)


def build_node(llm: ChatGroq):
    return make_specialist_node(AgentType.DOCS, SYSTEM_PROMPT, llm)
