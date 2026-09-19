"""Quality specialist. Asks 'is the logic right?' - the classic review
pass: real bugs, not style.
"""
from langchain_groq import ChatGroq

from pr_reviewer.agents.base_agent import make_specialist_node
from pr_reviewer.domain.contracts import AgentType

SYSTEM_PROMPT = (
    "You are a senior engineer reviewing a pull request diff for correctness. "
    "Look only for real bugs: logic errors, off-by-one mistakes, null/undefined "
    "handling, race conditions, resource leaks, incorrect error handling. Skip "
    "nitpicks and style comments a linter would already catch. If you find "
    "nothing, return an empty findings list."
)


def build_node(llm: ChatGroq):
    return make_specialist_node(AgentType.QUALITY, SYSTEM_PROMPT, llm)
