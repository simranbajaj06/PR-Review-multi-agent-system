"""Wires the four specialist agents and the aggregator into one LangGraph
graph. This is the only file that knows the topology - fan-out via Send,
join into aggregate. Every agent module only knows its own prompt; the
aggregator only knows how to merge Finding objects. Adding a 5th
specialist means adding one line to _AGENT_BUILDERS and nothing else.
"""
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from pr_reviewer.agents import docs_agent, quality_agent, security_agent, tests_agent
from pr_reviewer.domain.contracts import AgentType
from pr_reviewer.orchestrator.aggregator import aggregate_node
from pr_reviewer.orchestrator.state import ReviewState

_AGENT_BUILDERS = {
    AgentType.SECURITY: security_agent.build_node,
    AgentType.QUALITY: quality_agent.build_node,
    AgentType.TESTS: tests_agent.build_node,
    AgentType.DOCS: docs_agent.build_node,
}


def build_graph(llm: ChatGroq):
    def dispatch(state: ReviewState):
        # Fan out to all four specialists in parallel with the same state.
        # LangGraph waits for every branch to reach "aggregate" before that
        # node runs - the join is encoded in the graph, not hand-orchestrated.
        return [Send(agent_type.value, state) for agent_type in AgentType]

    graph = StateGraph(ReviewState)
    for agent_type, build_node in _AGENT_BUILDERS.items():
        graph.add_node(agent_type.value, build_node(llm))
        graph.add_edge(agent_type.value, "aggregate")
    graph.add_node("aggregate", aggregate_node)
    graph.add_conditional_edges(START, dispatch, [a.value for a in AgentType])
    graph.add_edge("aggregate", END)
    return graph.compile()
