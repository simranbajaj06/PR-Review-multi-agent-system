"""Security specialist. Asks 'could this be exploited?' - a distinct
mindset from correctness or test-coverage review, so it gets its own
prompt and its own pass over the diff rather than being folded into a
general-purpose review prompt.
"""
from langchain_groq import ChatGroq

from pr_reviewer.agents.base_agent import make_specialist_node
from pr_reviewer.domain.contracts import AgentType

SYSTEM_PROMPT = (
    "You are a security-focused senior engineer reviewing a pull request diff. "
    "Look only for security issues: injection risks, secrets in code, auth bypasses, "
    "unsafe deserialization, SSRF, path traversal, missing input validation on "
    "untrusted data. Ignore style, correctness, and test coverage - other reviewers "
    "cover those. If you find nothing, return an empty findings list."
)


def build_node(llm: ChatGroq):
    return make_specialist_node(AgentType.SECURITY, SYSTEM_PROMPT, llm)
