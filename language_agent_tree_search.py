# Language Agent Tree Search

# This is a tutorial from https://langchain-ai.github.io/langgraph/tutorials/lats/lats/ . For more details see the link and the paper https://arxiv.org/abs/2310.04406
# 
# Language Agent Tree Search (LATS), by Zhou, et. al, is a general LLM agent search algorithm that combines reflection/evaluation and search (specifically monte-carlo trees search) to get achieve better overall task performance compared to similar techniques like ReACT, Reflexion, or Tree of Thoughts.


# It has four main steps:
# 
# Select: pick the best next actions based on the aggregate rewards from step (2). Either respond (if a solution is found or the max search depth is reached) or continue searching.
# Expand and simulate: select the "best" 5 potential actions to take and execute them in parallel.
# Reflect + Evaluate: observe the outcomes of these actions and score the decisions based on reflection (and possibly external feedback)
# Backpropagate: update the scores of the root trajectories based on the outcomes.


import subprocess
import sys
import getpass
import os
import math
from collections import deque
from typing import Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool
from langgraph.prebuilt import ToolNode
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain as as_runnable
from typing import Literal
from langgraph.graph import END, StateGraph, START
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableConfig
from collections import defaultdict
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
# from langgraph.prebuilt import ToolNode


def install_packages(packages):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "--quiet", *package.split()])

install_packages([
    "langchain langgraph langchain_openai",
    "tavily-python",
    "langchain langchain-community",
    "duckduckgo-search"
])


# For web search you can use different services. In the original tutorial, Tavily API is used, which requires a MY_TELEMETRY_KEY at https://app.tavily.com/home. It has limited free credits, so currently DuckDuckGo service, which is free but worse, is implemented later.

# Update OpenAI API key

def _set_if_undefined(var_name: str, secret_value: str) -> None:
    if os.environ.get(var_name):
        return
    os.environ[var_name] = secret_value
### Insert your OPENAI_API_KEY and MY_TELEMETRY_KEY

_set_if_undefined("OPENAI_API_KEY", "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") # insert key in the last string
#_set_if_undefined("MY_TELEMETRY_KEY", "tvlV0lbT") # insert key in the last string

#os.environ["TAVILY_API_KEY"] = "tvly-d4V0lbT" # insert key in the last string



# # Graph State

class Reflection(BaseModel):
    reflections: str = Field(
        description="The critique and reflections on the sufficiency, superfluency,"
        " and general quality of the response"
    )
    score: int = Field(
        description="Score from 0-10 on the quality of the candidate response.",
        gte=0,
        lte=10,
    )
    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task."
    )

    def as_message(self):
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0


class Node:
    def __init__(
        self,
        messages: list[BaseMessage],
        reflection: Reflection,
        parent: Optional["Node"] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution if reflection else False
        if self._is_solved:
            self._mark_tree_as_solved()
        self.backpropagate(reflection.normalized_score)

    def __repr__(self) -> str:
        return (
            f"<Node value={self.value}, visits={self.visits},"
            f" solution={self.messages} reflection={self.reflection}/>"
        )

    @property
    def is_solved(self):
        """If any solutions exist, we can end the search."""
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def best_child_score(self):
        """Return the child with the highest value."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True):
        if include_reflections:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        """Get messages representing this search branch."""
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_messages(include_reflections=include_reflections)[::-1]
            )
            node = node.parent
        # Reverse the final back-tracked trajectory to return in the correct order
        return messages[::-1]  # root solution, reflection, child 1, ...

    def _get_all_children(self):
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes

    def get_best_solution(self):
        """Return the best solution from within the current sub-tree."""
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            # We filter out all non-terminal, non-solution trajectories
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self):
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent


# The graph state itself


class TreeState(TypedDict):
    # The full tree
    root: Node
    # The original input
    input: str


# # Define Language Agent

llm = ChatOpenAI(model="gpt-4o")


# Tavily search engine - API, limited quantity of free credits.

# search = TavilySearchAPIWrapper()
# tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
# tools = [tavily_tool]
# tool_node = ToolNode(tools=tools)


# Duckduckgo search engine, it is worse but completely free.

# Instantiate the API wrapper
search = DuckDuckGoSearchAPIWrapper()

# Wrap it into a LangChain Tool manually
duckduckgo_tool = Tool(
    name="duckduckgo_search",
    description="Use this tool to search the web using DuckDuckGo.",
    func=search.run
)

# Create the tool node
tools = [duckduckgo_tool]
tool_node = ToolNode(tools=tools)


# Reflection

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Reflect and grade the assistant response to the user question below.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="candidate"),
    ]
)

reflection_llm_chain = (
    prompt
    | llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(
        run_name="Reflection"
    )
    | PydanticToolsParser(tools=[Reflection])
)


@as_runnable
def reflection_chain(inputs) -> Reflection:
    tool_choices = reflection_llm_chain.invoke(inputs)
    reflection = tool_choices[0]
    if not isinstance(inputs["candidate"][-1], AIMessage):
        reflection.found_solution = False
    return reflection


# Initial Response

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)


initial_answer_chain = prompt_template | llm.bind_tools(tools=tools).with_config(
    run_name="GenerateInitialCandidate"
)

parser = JsonOutputToolsParser(return_id=True)



# Starting Node

# Define the node we will add to the graph
def generate_initial_response(state: TreeState) -> dict:
    """Generate the initial candidate response."""
    res = initial_answer_chain.invoke({"input": state["input"]})
    parsed = parser.invoke(res)
    tool_responses = [
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": r["type"], "args": r["args"], "id": r["id"]}
                        ],
                    )
                ]
            }
        )
        for r in parsed
    ]
    output_messages = [res] + [tr["messages"][0] for tr in tool_responses]
    reflection = reflection_chain.invoke(
        {"input": state["input"], "candidate": output_messages}
    )
    root = Node(output_messages, reflection=reflection)
    return {
        **state,
        "root": root,
    }


# # Candidate Generation

# This generates N candidate values
# for a single input to sample actions from the environment


def generate_candidates(messages: ChatPromptValue, config: RunnableConfig):
    n = config["configurable"].get("N", 5)
    bound_kwargs = llm.bind_tools(tools=tools).kwargs
    chat_result = llm.generate(
        [messages.to_messages()],
        n=n,
        callbacks=config["callbacks"],
        run_name="GenerateCandidates",
        **bound_kwargs,
    )
    return [gen.message for gen in chat_result.generations[0]]


expansion_chain = prompt_template | generate_candidates


def select(root: Node) -> dict:
    """Starting from the root node a child node is selected at each tree level until a leaf node is reached."""

    if not root.children:
        return root

    node = root
    while node.children:
        max_child = max(node.children, key=lambda child: child.upper_confidence_bound())
        node = max_child

    return node


def expand(state: TreeState, config: RunnableConfig) -> dict:
    """Starting from the "best" node in the tree, generate N candidates for the next step."""
    root = state["root"]
    best_candidate: Node = select(root)
    messages = best_candidate.get_trajectory()
    # Generate N candidates from the single child candidate
    new_candidates = expansion_chain.invoke(
        {"input": state["input"], "messages": messages}, config
    )
    parsed = parser.batch(new_candidates)
    flattened = [
        (i, tool_call)
        for i, tool_calls in enumerate(parsed)
        for tool_call in tool_calls
    ]
    tool_responses = [
        (
            i,
            tool_node.invoke(
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": tool_call["type"],
                                    "args": tool_call["args"],
                                    "id": tool_call["id"],
                                }
                            ],
                        )
                    ]
                }
            ),
        )
        for i, tool_call in flattened
    ]
    collected_responses = defaultdict(list)
    for i, resp in tool_responses:
        collected_responses[i].append(resp["messages"][0])
    output_messages = []
    for i, candidate in enumerate(new_candidates):
        output_messages.append([candidate] + collected_responses[i])

    # Reflect on each candidate
    # For tasks with external validation, you'd add that here.
    reflections = reflection_chain.batch(
        [{"input": state["input"], "candidate": msges} for msges in output_messages],
        config,
    )
    # Grow tree
    child_nodes = [
        Node(cand, parent=best_candidate, reflection=reflection)
        for cand, reflection in zip(output_messages, reflections)
    ]
    best_candidate.children.extend(child_nodes)
    # We have already extended the tree directly, so we just return the state
    return state


# # Create Graph

def should_loop(state: TreeState):
    """Determine whether to continue the tree search."""
    root = state["root"]
    if root.is_solved:
        return END
    if root.height > 5:
        return END
    return "expand"


builder = StateGraph(TreeState)
builder.add_node("start", generate_initial_response)
builder.add_node("expand", expand)
builder.add_edge(START, "start")


builder.add_conditional_edges(
    "start",
    # Either expand/rollout or finish
    should_loop,
    ["expand", END],
)
builder.add_conditional_edges(
    "expand",
    # Either continue to rollout or finish
    should_loop,
    ["expand", END],
)

graph = builder.compile()

#from IPython.display import Image

#Image(graph.get_graph().draw_mermaid_png())


# # Example 1
# Invoke the model

def tree_search_answer(stat_prompt):

    question = stat_prompt 
    last_step = None
    for step in graph.stream({"input": question}):
        last_step = step
        step_name, step_state = next(iter(step.items()))
        print(step_name)
        print("rolled out: ", step_state["root"].height)
        print("---")
    
    solution_node = last_step["expand"]["root"].get_best_solution()
    best_trajectory = solution_node.get_trajectory(include_reflections=False)
    return best_trajectory[-1].content





