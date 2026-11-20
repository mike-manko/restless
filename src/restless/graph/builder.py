from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from restless.graph.context import RunContext
from restless.graph.nodes import AgentState, GraphInput, call_tools, check_budget, check_tool_calls, hard_failure_result, judge_result, load_scenario, next_executor_action

type RunGraph = CompiledStateGraph[AgentState, RunContext, GraphInput, AgentState]


# Wire the nodes into the run's graph. Compile once and reuse it for every scenario:
def build_graph() -> RunGraph:
    """load_scenario -> executor <-> tools -> judge or hard failure. See design.md, "LangGraph graph"."""

    builder = StateGraph(AgentState, context_schema=RunContext, input_schema=GraphInput)

    builder.add_node("load_scenario", load_scenario)
    builder.add_node("next_executor_action", next_executor_action)
    builder.add_node("call_tools", call_tools)
    builder.add_node("judge_result", judge_result)
    builder.add_node("hard_failure_result", hard_failure_result)

    builder.add_edge(START, "load_scenario")
    builder.add_edge("load_scenario", "next_executor_action")

    builder.add_conditional_edges("next_executor_action", check_tool_calls)
    builder.add_conditional_edges("call_tools", check_budget)

    builder.add_edge("judge_result", END)
    builder.add_edge("hard_failure_result", END)

    return builder.compile()
