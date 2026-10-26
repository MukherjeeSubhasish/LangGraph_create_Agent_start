# LangGraph Components — Anthropic (Claude) version
#
# Setup:
#   uv pip install langgraph langchain-anthropic langchain-tavily python-dotenv
#   .env file:
#     ANTHROPIC_API_KEY=sk-ant-...
#     TAVILY_API_KEY=tvly-...        # free key at https://app.tavily.com

# %%
import operator
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch   # replaces deprecated langchain_community TavilySearchResults

_ = load_dotenv()

MODEL = "claude-sonnet-5-5"  

tool = TavilySearch(max_results=4)
print(type(tool))
print(tool.name)


# %% State + Agent
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        return len(state["messages"][-1].tool_calls) > 0

    def call_llm(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        return {"messages": [self.model.invoke(messages)]}

    def take_action(self, state: AgentState):
        results = []
        for t in state["messages"][-1].tool_calls:
            print(f"Calling: {t}")
            if t["name"] not in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            else:
                result = self.tools[t["name"]].invoke(t["args"])
            results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
        print("Back to the model!")
        return {"messages": results}


prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""


model = ChatAnthropic(model=MODEL, max_tokens=2048)
abot = Agent(model, [tool], system=prompt)

# %% Save graph picture (IPython Image doesn't display in a plain script)
try:
    with open("graph.png", "wb") as f:
        f.write(abot.graph.get_graph().draw_mermaid_png())  # no graphviz install needed
    print("Saved graph.png")
except Exception as e:
    print("Could not draw graph:", e)
    print(abot.graph.get_graph().draw_ascii())  # needs: uv pip install grandalf

# %% Single question
result = abot.graph.invoke({"messages": [HumanMessage(content="What is the weather in sf?")]})
print(result["messages"][-1].text)  # .text handles Claude's list-of-blocks content

# %% Parallel tool calls
result = abot.graph.invoke({"messages": [HumanMessage(content="What is the weather in SF and LA?")]})
print(result["messages"][-1].text)

# %% Multi-step research
query = (
    "Who won the super bowl in 2024? In what state is the winning team headquarters located? "
    "What is the GDP of that state? Answer each question."
)
result = abot.graph.invoke({"messages": [HumanMessage(content=query)]})
print(result["messages"][-1].text)