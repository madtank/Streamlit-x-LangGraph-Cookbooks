from typing import Annotated, TypedDict, Literal, Any

from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, StructuredTool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_aws import ChatBedrockConverse
import streamlit as st

# Define a search tool using DuckDuckGo API wrapper
search_DDG = StructuredTool.from_function(
        name="Search",
        func=DuckDuckGoSearchAPIWrapper().run,  # Executes DuckDuckGo search using the provided query
        description=f"""
        useful for when you need to answer questions about current events. You should ask targeted questions
        """,
    )

# Define a function to evaluate Python code
def evaluate_python_code(code: str) -> Any:
    try:
        # Execute the Python code
        namespace = {}
        exec(code, namespace)
        return namespace
    except Exception as e:
        return f"Error: {str(e)}"

# Call to evaluate Python code
@tool
def python_repl(code: str) -> Any:
    """
    Evaluate the provided Python code and return the result.

    Args:
        code (str): The Python code to evaluate.

    Returns:
        Any: The result of the evaluated code or an error message.
    """
    result = evaluate_python_code(code)
    return result

# Define a function to render Markdown text
def render_markdown(text: str) -> str:
    # You can use a Markdown rendering library here to render the Markdown text
    # For simplicity, let's just return the text as is for now
    return text

# Call to render Markdown text
@tool
def markdown_tool(text: str) -> str:
    """
    Render the provided Markdown text and return the formatted output.

    Args:
        text (str): The Markdown text to render.

    Returns:
        str: The formatted output after rendering the Markdown text.
    """
    formatted_text = render_markdown(text)
    return formatted_text

# List of tools that will be accessible to the graph via the ToolNode
tools = [search_DDG, python_repl, markdown_tool]
tool_node = ToolNode(tools)

# This is the default state same as "MessageState" TypedDict but allows us accessibility to custom keys
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Custom keys for additional data can be added here such as - conversation_id: str

graph = StateGraph(GraphsState)

# Function to decide whether to continue tool usage or end the process
def should_continue(state: GraphsState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:  # Check if the last message has any tool calls
        return "tools"  # Continue to tool execution
    return "__end__"  # End the conversation if no tool is needed

# Core invocation of the model
def _call_model(state: GraphsState):
    messages = state["messages"]
    llm = ChatBedrockConverse(
        model="us.meta.llama3-2-90b-instruct-v1:0",
        temperature=0.3
    ).bind_tools(tools)
    response = llm.invoke(messages)
    return {"messages": [response]}  # add the response to the messages using LangGraph reducer paradigm

# Define the structure (nodes and directional edges between nodes) of the graph
graph.add_edge(START, "modelNode")
graph.add_node("tools", tool_node)
graph.add_node("modelNode", _call_model)

# Add conditional logic to determine the next step based on the state (to continue or to end)
graph.add_conditional_edges(
    "modelNode",
    should_continue,  # This function will decide the flow of execution
)
graph.add_edge("tools", "modelNode")

# Compile the state graph into a runnable object
graph_runnable = graph.compile()