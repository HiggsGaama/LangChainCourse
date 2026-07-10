---
layout: default
title: LangGraph Deep Dive
---

# Module 9 — LangGraph Deep Dive

[Module 8 — Agents with LangGraph](08-agents-with-langgraph.md) showed you how to *use* LangGraph's prebuilt agent (`create_react_agent`). This module pulls that abstraction apart. You will build graphs by hand: typed state, plain-function nodes, edges that branch and loop, persistence, human-in-the-loop interrupts, time travel, and fine-grained streaming.

If [LCEL & the Runnable Interface](04-lcel-and-runnables.md) taught you to compose **acyclic data pipelines**, this module teaches you to build **stateful, cyclic, durable workflows**. LangGraph is the runtime that powers production agents — understanding it deeply is the difference between gluing prebuilts together and engineering systems you can debug, resume, and trust.

```bash
pip install -U langgraph langchain-anthropic langgraph-checkpoint-sqlite
```

---

## 1. The mental model

LangGraph is a small, opinionated state machine library. Three concepts carry everything:

1. **State** — a single, typed data structure shared by every node. Think of it as the "world" the graph operates on.
2. **Nodes** — plain Python functions that read the current state and return a **partial update** (a dict of the keys they changed).
3. **Edges** — wiring that decides which node runs next. Edges can be static (`A → B`) or **conditional** (a router function chooses the next node at runtime, enabling branches and **cycles**).

A fourth concept glues updates together: **reducers** (also called channels). When two nodes write the same state key, a reducer decides how the writes merge. The default reducer overwrites; you can choose append, or write your own.

```mermaid
flowchart LR
    START((START)) --> A[node: agent]
    A -- has tool calls --> T[node: tools]
    A -- done --> END((END))
    T --> A
    subgraph State["Shared typed State"]
      direction TB
      M["messages: list  reducer=add_messages"]
      C["counter: int  reducer=overwrite"]
    end
```

### LCEL vs LangGraph

| | **LCEL (`Runnable` / pipe)** | **LangGraph (`StateGraph`)** |
|---|---|---|
| Topology | Directed **acyclic** graph (a pipe) | Arbitrary graph: **branches and cycles** |
| State | Data flows through; no shared mutable state | Single shared, typed `State` object |
| Control flow | Fixed at build time | **Dynamic** — router functions decide at runtime |
| Persistence | None built in | **Checkpointers** snapshot state per step |
| Human-in-the-loop | Not native | **First-class** via `interrupt()` |
| Best for | Prompt → model → parse; RAG retrieval chains | Agents, multi-step workflows, anything that loops or pauses |

> **Note:** A compiled LangGraph graph *is itself a `Runnable`*. You can drop a graph into an LCEL chain, and you can call LCEL chains from inside graph nodes. They compose. Use LCEL for the straight-line bits; use LangGraph when you need cycles, memory, or pauses.

---

## 2. State: the typed core

State is the contract every node shares. LangGraph reads the type annotations to learn which **channels** exist and which **reducer** governs each.

### 2.1 `TypedDict` — the common case

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Annotated[<type>, <reducer>]: add_messages appends and dedups by id.
    messages: Annotated[list[AnyMessage], add_messages]
    # No reducer -> default behavior: each write OVERWRITES the previous value.
    step_count: int
```

The `Annotated[T, reducer]` pattern is the heart of state design. The first element is the static type; the second is a **callable `(old, new) -> merged`** that LangGraph applies when a node writes that key.

### 2.2 Reducers: how updates merge

```python
import operator
from typing import Annotated, TypedDict


class ReducerDemo(TypedDict):
    overwritten: str                          # default reducer -> last write wins
    appended: Annotated[list[int], operator.add]   # list concatenation
    # add_messages is a smarter list reducer specialized for chat messages.
```

- **Default (no annotation):** each node's value **replaces** whatever was there. Good for scalars and "current best answer" fields.
- **`operator.add`:** concatenates lists (or sums numbers / merges strings). Essential when multiple parallel nodes contribute to one list (see [§5, Send](#5-the-send-api--map-reduce)).
- **`add_messages`** (from `langgraph.graph.message`): appends messages **and deduplicates by message `id`**, so re-running a node that re-emits a message won't double it. It also handles `RemoveMessage` to delete history. This is why every chat agent uses it.

> **⚠️ Gotcha:** With the **default** reducer, two nodes writing the same key *in the same super-step* (e.g. parallel branches) is an error — LangGraph cannot pick a winner. If parallel nodes must write the same key, give it an aggregating reducer like `operator.add`.

### 2.3 Custom reducers

A reducer is just a function. Here is one that keeps a bounded sliding window of the last N messages:

```python
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def add_messages_windowed(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    """Append like add_messages, then keep only the most recent 20 messages."""
    merged = add_messages(left, right)
    return merged[-20:]
```

Wire it in exactly like any other reducer: `messages: Annotated[list[AnyMessage], add_messages_windowed]`.

> **✅ Best practice:** Choose the reducer when you design the schema, not as an afterthought. The reducer *is* the concurrency model of your graph.

### 2.4 `MessagesState` shortcut

The "list of messages with `add_messages`" pattern is so common LangGraph ships it:

```python
from langgraph.graph import MessagesState


class State(MessagesState):
    # Inherits: messages: Annotated[list[AnyMessage], add_messages]
    # Add your own channels alongside it:
    remaining_steps: int
    user_id: str
```

### 2.5 Pydantic and dataclass state

`TypedDict` gives you typing without runtime validation. If you want **validation on every node return**, use a Pydantic model; for lightweight mutability semantics, a `dataclass` works too.

```python
from typing import Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class PydanticState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    score: float = 0.0  # validated as a float at every node boundary
```

> **Note:** Pydantic state validates the **input** to nodes. Validation has a per-step cost and stricter coercion rules; reach for it when malformed state is a real risk (e.g. nodes built from external/untrusted data). `TypedDict` is the default for most graphs.

### 2.6 Input/output schemas and private channels

By default, the input, output, and internal state all share one schema. You can separate them: expose a clean public input/output while keeping scratch fields internal.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class InputState(TypedDict):
    question: str


class OutputState(TypedDict):
    answer: str


class OverallState(TypedDict):
    question: str
    answer: str
    scratch: str          # private: never appears in input or output


def think(state: OverallState) -> dict:
    return {"scratch": f"reasoning about: {state['question']}"}


def respond(state: OverallState) -> dict:
    return {"answer": f"Based on {state['scratch']!r}, here is the answer."}


builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
builder.add_node("think", think)
builder.add_node("respond", respond)
builder.add_edge(START, "think")
builder.add_edge("think", "respond")
builder.add_edge("respond", END)
graph = builder.compile()

print(graph.invoke({"question": "What is LangGraph?"}))
# -> {'answer': "Based on 'reasoning about: What is LangGraph?', here is the answer."}
# Note: 'scratch' and 'question' are filtered out of the OUTPUT.
```

> **⚠️ Verify:** The constructor keyword names have evolved across versions. Modern LangGraph accepts `input_schema=` / `output_schema=`; older code used `input=` / `output=`. If your version errors, check `StateGraph.__init__`'s signature.

---

## 3. The `StateGraph` API

The build-then-compile lifecycle:

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def greet(state: State) -> dict:
    # A node RECEIVES the full state and RETURNS a partial update (only changed keys).
    last = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"You said: {last}")]}


builder = StateGraph(State)        # 1. construct over a schema
builder.add_node("greet", greet)   # 2. register nodes (name -> callable)
builder.add_edge(START, "greet")   # 3. wire edges (START is the virtual entry)
builder.add_edge("greet", END)     # END is the virtual exit
graph = builder.compile()          # 4. compile -> a runnable CompiledStateGraph

result = graph.invoke({"messages": [HumanMessage("hello")]})
print(result["messages"][-1].content)   # -> "You said: hello"
```

Key facts:

- **`add_node(name, fn)`** — `fn` can be sync or async. The node name is how edges refer to it. (You can also pass just `fn` and the function name is used.)
- **`add_edge(src, dst)`** — unconditional. Use `START`/`END` (imported from `langgraph.graph`) as the virtual entry/exit. `add_edge(START, "x")` is the modern form of the older `set_entry_point("x")`.
- **Nodes return PARTIAL updates.** Return `{"messages": [...]}` to touch only `messages`; every other channel is left untouched. Returning `{}` or `None` changes nothing.
- **`compile()`** returns a `CompiledStateGraph`, which implements the full `Runnable` interface: `.invoke`, `.stream`, `.ainvoke`, `.astream`, `.batch`, plus graph-specific methods like `.get_state`.

> **🔧 Try it:** Call `print(graph.get_graph().draw_mermaid())` on any compiled graph to print a Mermaid diagram of its topology. Paste it into a Markdown file to visualize your own graphs.

---

## 4. Conditional edges & cycles — building the agent loop by hand

`create_react_agent` hides a loop. Here it is, explicit. A **router function** inspects state and returns the *name* of the next node (or `END`). Returning a node name that points back upstream creates a **cycle**.

```python
from typing import Annotated, TypedDict, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"It's 18°C and clear in {city}."


tools = [get_weather]
llm = ChatAnthropic(model="claude-sonnet-4-6").bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def agent(state: State) -> dict:
    """Call the model; it may respond directly or request tool calls."""
    return {"messages": [llm.invoke(state["messages"])]}


def route(state: State) -> Literal["tools", "__end__"]:
    """Router: if the last AI message asked for tools, go to 'tools'; else finish."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END   # END == "__end__"


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))   # prebuilt: executes tool calls -> ToolMessages
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")           # <-- the cycle: tools loops back to agent
graph = builder.compile()

for msg in graph.invoke({"messages": [HumanMessage("Weather in Paris?")]})["messages"]:
    msg.pretty_print()
```

`add_conditional_edges(source, router_fn, mapping)`:

- **`source`** — the node whose completion triggers the decision.
- **`router_fn`** — receives state, returns one or more keys.
- **`mapping`** — `{router_return_value: destination_node}`. The mapping is optional when `router_fn` returns actual node names directly, but supplying it keeps the topology explicit and improves the rendered diagram.

### `tools_condition` — the prebuilt router

The "did the model request tools?" check is so standard LangGraph ships it. The router above is exactly equivalent to:

```python
from langgraph.prebuilt import tools_condition

# Returns "tools" when the last message has tool_calls, otherwise END.
builder.add_conditional_edges("agent", tools_condition)
```

> **✅ Best practice:** A router function should be **pure and cheap** — read state, return a string. Never call a model or do I/O inside a router; put that work in a node so it shows up as a step in the trace and is covered by checkpointing.

---

## 5. The `Send` API — map-reduce / dynamic fan-out

Sometimes you don't know until runtime *how many* parallel branches you need — e.g. "generate a section for each of these N topics." A router can return a **list of `Send` objects**, each dispatching the same (or a different) node with a *custom, per-branch state*. Results merge back through a reducer.

```python
import operator
from typing import Annotated, TypedDict
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

llm = ChatAnthropic(model="claude-sonnet-4-6")


class OverallState(TypedDict):
    topic: str
    subjects: list[str]
    # Parallel workers all append here -> operator.add merges their lists safely.
    jokes: Annotated[list[str], operator.add]


class JokeState(TypedDict):
    # The per-branch state passed to each worker. Does NOT need to match OverallState.
    subject: str


def generate_subjects(state: OverallState) -> dict:
    resp = llm.invoke(f"List 3 comma-separated subtopics of {state['topic']}. Only the list.")
    subjects = [s.strip() for s in resp.content.split(",")][:3]
    return {"subjects": subjects}


def continue_to_jokes(state: OverallState) -> list[Send]:
    # FAN OUT: one Send per subject -> N parallel invocations of "write_joke".
    return [Send("write_joke", {"subject": s}) for s in state["subjects"]]


def write_joke(state: JokeState) -> dict:
    resp = llm.invoke(f"Write a one-line joke about {state['subject']}.")
    return {"jokes": [resp.content]}   # appended via operator.add


builder = StateGraph(OverallState)
builder.add_node("generate_subjects", generate_subjects)
builder.add_node("write_joke", write_joke)
builder.add_edge(START, "generate_subjects")
builder.add_conditional_edges("generate_subjects", continue_to_jokes, ["write_joke"])
builder.add_edge("write_joke", END)
graph = builder.compile()

result = graph.invoke({"topic": "databases", "subjects": [], "jokes": []})
for j in result["jokes"]:
    print("-", j)
# - Three jokes, generated in parallel, collected into 'jokes'.
```

```mermaid
flowchart TD
    START((START)) --> G[generate_subjects]
    G -. "Send x3" .-> W1[write_joke: subject A]
    G -. "Send x3" .-> W2[write_joke: subject B]
    G -. "Send x3" .-> W3[write_joke: subject C]
    W1 --> R["jokes (operator.add)"]
    W2 --> R
    W3 --> R
    R --> END((END))
```

> **⚠️ Gotcha:** The state object inside a `Send(node, state)` is **independent** of the overall schema — it's whatever that node expects. The merge back into `OverallState` happens only through the **reducer** on the shared key. Without `operator.add` on `jokes`, the parallel writes would collide.

---

## 6. Subgraphs

A compiled graph is a `Runnable`, so you can use one **as a node** inside another graph. This is how you build modular, reusable, independently testable components (e.g. a "research" subgraph reused inside several agents).

There are two wiring modes:

**(a) Shared state schema** — parent and subgraph share channel names; add the compiled subgraph directly as a node:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    value: int


def double(state: State) -> dict:
    return {"value": state["value"] * 2}


sub = StateGraph(State)
sub.add_node("double", double)
sub.add_edge(START, "double")
sub.add_edge("double", END)
subgraph = sub.compile()

parent = StateGraph(State)
parent.add_node("subgraph", subgraph)   # a compiled graph used directly as a node
parent.add_edge(START, "subgraph")
parent.add_edge("subgraph", END)
app = parent.compile()

print(app.invoke({"value": 5}))   # -> {'value': 10}
```

**(b) Different schemas** — wrap the subgraph in a function node that **transforms** parent state into the subgraph's input and maps results back:

```python
def call_subgraph(state: ParentState) -> dict:
    sub_input = {"value": state["count"]}          # parent -> subgraph
    sub_result = subgraph.invoke(sub_input)
    return {"count": sub_result["value"]}          # subgraph -> parent
```

> **✅ Best practice:** Use a subgraph when a chunk of logic is reusable, has its own internal control flow (loops/branches), or you want to test it in isolation. Use the shared-schema form for tightly coupled stages; use the wrapper form when the subgraph is a black box with its own vocabulary. Subgraph steps appear nested in LangSmith traces and in `stream(..., subgraphs=True)`.

---

## 7. Persistence & checkpointers

Pass a **checkpointer** at compile time and LangGraph snapshots the full state **after every super-step**, keyed by a `thread_id`. This single feature unlocks: conversation memory across calls, crash recovery (durable execution), human-in-the-loop, and time travel.

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "user-42"}}

# Turn 1
graph.invoke({"messages": [HumanMessage("My name is Sam.")]}, config)
# Turn 2 — same thread_id -> prior messages are loaded automatically from the checkpoint.
out = graph.invoke({"messages": [HumanMessage("What's my name?")]}, config)
print(out["messages"][-1].content)   # -> "Your name is Sam."
```

A **thread** is one conversation/session. Same `thread_id` ⇒ continuation. New `thread_id` ⇒ fresh state. There is no global state — everything is per-thread.

### Checkpointer backends

| Backend | Import | Package | Use for |
|---|---|---|---|
| `MemorySaver` | `langgraph.checkpoint.memory` | built into `langgraph` | tests, notebooks, demos (lost on restart) |
| `SqliteSaver` | `langgraph.checkpoint.sqlite` | `langgraph-checkpoint-sqlite` | single-process apps, local persistence |
| `PostgresSaver` | `langgraph.checkpoint.postgres` | `langgraph-checkpoint-postgres` | **production**, multi-replica deployments |

```python
# SQLite — note the context-manager form
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({"messages": [HumanMessage("hi")]}, {"configurable": {"thread_id": "1"}})

# Postgres — run checkpointer.setup() once to create tables
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:pass@localhost:5432/mydb?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()   # first run only: creates the checkpoint tables
    graph = builder.compile(checkpointer=checkpointer)
    ...
```

> **Note:** Async graphs (`.ainvoke` / `.astream`) need async checkpointers: `AsyncSqliteSaver` (`langgraph.checkpoint.sqlite.aio`) and `AsyncPostgresSaver` (`langgraph.checkpoint.postgres.aio`). Mixing a sync saver into an async run will raise.

### Inspecting and editing state

```python
config = {"configurable": {"thread_id": "user-42"}}

snapshot = graph.get_state(config)
print(snapshot.values)        # current channel values
print(snapshot.next)          # tuple of node(s) that will run next ('()' means done)
print(snapshot.config)        # includes the checkpoint_id of THIS snapshot

# Full audit trail — newest first. Each entry is a StateSnapshot.
for snap in graph.get_state_history(config):
    print(snap.config["configurable"]["checkpoint_id"], "->", snap.next)

# Manually patch state (writes a new checkpoint; respects reducers).
graph.update_state(config, {"messages": [HumanMessage("(injected) please be concise")]})
```

> **⚠️ Gotcha:** `get_state` / `get_state_history` / `update_state` **require a checkpointer**. Without one, there is no state to read or a history to walk — these methods will error.

---

## 8. Human-in-the-loop, deeply

Production agents frequently need a human to approve an action, supply a missing value, or correct a wrong turn before the graph proceeds. LangGraph supports this through **dynamic interrupts** (the modern, preferred way) and **static interrupts** (breakpoints).

> **Note:** Human-in-the-loop **requires a checkpointer** — pausing means persisting state and resuming later, possibly in a different process.

### 8.1 Dynamic interrupts: `interrupt()` + `Command(resume=...)`

Call `interrupt(payload)` inside a node. Execution pauses; `payload` surfaces to the caller. The caller inspects it, then resumes by invoking the graph again with `Command(resume=value)`. The `interrupt()` call then *returns that value* and the node continues.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


class State(TypedDict):
    draft: str
    approved: bool


def write_draft(state: State) -> dict:
    return {"draft": "Dear customer, your refund has been processed."}


def human_approval(state: State) -> dict:
    # Pauses here. The dict is surfaced to the caller for review.
    decision = interrupt({"draft": state["draft"], "action": "approve_or_edit"})
    # Resumes HERE when Command(resume=...) is sent; `decision` is that resume value.
    if decision["type"] == "approve":
        return {"approved": True}
    return {"draft": decision["edited_text"], "approved": True}


def send(state: State) -> dict:
    print(f"SENDING: {state['draft']}")
    return {}


builder = StateGraph(State)
builder.add_node("write_draft", write_draft)
builder.add_node("human_approval", human_approval)
builder.add_node("send", send)
builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "human_approval")
builder.add_edge("human_approval", "send")
builder.add_edge("send", END)
graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "ticket-7"}}

# First run: pauses at human_approval.
result = graph.invoke({"draft": "", "approved": False}, config)
print(result["__interrupt__"])   # the surfaced payload — show this to your reviewer

# ----- (later, possibly a different HTTP request) -----

# Resume with an edit:
final = graph.invoke(
    Command(resume={"type": "edit", "edited_text": "Dear Sam, your refund is on its way."}),
    config,
)
# -> SENDING: Dear Sam, your refund is on its way.
```

```mermaid
sequenceDiagram
    participant C as Caller / UI
    participant G as Graph
    C->>G: invoke(input, thread_id)
    G->>G: write_draft
    G->>G: human_approval -> interrupt(payload)
    G-->>C: pause; surface __interrupt__ payload
    Note over C: human reviews / edits
    C->>G: invoke(Command(resume=value), thread_id)
    G->>G: human_approval resumes, returns value
    G->>G: send
    G-->>C: final state
```

> **⚠️ Gotcha — nodes re-run from the top on resume.** When you resume, the **entire node** containing the `interrupt()` runs again from its first line; `interrupt()` then returns the resume value instead of pausing. So any code **before** `interrupt()` executes twice. Keep pre-interrupt code side-effect-free (no DB writes, no emails) — do side effects *after* the interrupt, or in a downstream node.

### 8.2 Static interrupts (breakpoints)

Pause **before** or **after** a named node without changing node code — useful for debugging or a blanket "always confirm before tools" policy.

```python
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["send"],     # pause right before 'send' runs
    # interrupt_after=["write_draft"],
)

config = {"configurable": {"thread_id": "ticket-9"}}
graph.invoke({"draft": "", "approved": False}, config)   # stops before 'send'

# Inspect, optionally edit, then continue by invoking with None:
graph.invoke(None, config)   # resumes from the breakpoint
```

### 8.3 The three core patterns

- **Approve / reject:** `interrupt()` returns a boolean-ish decision; the router sends to `proceed` or `cancel`.
- **Edit state:** surface the proposed value, resume with a corrected one, write it back (the refund-edit example above).
- **Branch / route:** surface options, let the human pick the next node; the resumed value drives a conditional edge.

> **✅ Best practice:** Use **dynamic `interrupt()`** for product features (approvals, clarifications) — the pause point lives in the node logic where it belongs. Reserve **static `interrupt_before/after`** for debugging and operational breakpoints. Always pair interrupts with a persistent checkpointer in production so a pause can outlive the process.

---

## 9. Time travel

Because every super-step is checkpointed, you can **replay** the graph from any past checkpoint, or **fork** an alternate timeline from it. This is invaluable for debugging ("what if the model had chosen differently at step 3?") and for letting users branch a conversation.

```python
config = {"configurable": {"thread_id": "user-42"}}

# 1. Walk history (newest first) and pick a checkpoint to return to.
history = list(graph.get_state_history(config))
for i, snap in enumerate(history):
    print(i, snap.config["configurable"]["checkpoint_id"], snap.next)

target = history[2]   # some earlier StateSnapshot

# 2. REPLAY: re-run forward from that exact checkpoint by passing its full config.
#    Steps already computed before this point are re-used from the checkpoint (not recomputed).
for chunk in graph.stream(None, target.config, stream_mode="values"):
    print(chunk)

# 3. FORK: edit state at that checkpoint -> update_state returns a NEW checkpoint config.
#    Running from it creates a branching timeline without destroying the original.
forked_config = graph.update_state(
    target.config,
    {"messages": [HumanMessage("Actually, answer in French.")]},
)
graph.invoke(None, forked_config)   # proceeds down the alternate branch
```

```mermaid
flowchart LR
    c0[ckpt 0] --> c1[ckpt 1] --> c2[ckpt 2] --> c3[ckpt 3]
    c2 -. "update_state at ckpt 2" .-> f1[forked ckpt]
    f1 --> f2[alt continuation]
```

> **Note:** Passing a config that contains a specific `checkpoint_id` tells LangGraph *where to start*. Passing input `None` means "resume from the saved state" rather than "inject new input."

---

## 10. Streaming modes in detail

`CompiledStateGraph.stream(input, config, stream_mode=...)` (and async `.astream`) exposes execution progressively. The mode determines what each yielded chunk contains.

| `stream_mode` | Each chunk is… | Use for |
|---|---|---|
| `"values"` | the **full state** after each step | progress UIs that re-render the whole state |
| `"updates"` | just the **partial update** each node returned (`{node_name: update}`) | logging which node did what |
| `"messages"` | `(message_chunk, metadata)` — **LLM tokens** as they generate | token-by-token chat UIs |
| `"custom"` | arbitrary data a node emits via the stream writer | progress bars, tool sub-steps |
| `"debug"` | exhaustive checkpoint + task events | deep debugging |

```python
# updates: see each node's contribution
for chunk in graph.stream({"messages": [HumanMessage("Weather in Oslo?")]}, config,
                          stream_mode="updates"):
    print(chunk)
# {'agent': {'messages': [AIMessage(... tool_calls ...)]}}
# {'tools': {'messages': [ToolMessage('It's 18°C ...')]}}
# {'agent': {'messages': [AIMessage('It is 18°C and clear in Oslo.')]}}

# messages: stream LLM tokens (the second tuple element is metadata incl. node name)
for token, meta in graph.stream({"messages": [HumanMessage("Tell me a story")]}, config,
                                stream_mode="messages"):
    if token.content:
        print(token.content, end="", flush=True)
```

### Custom data with the stream writer

A node can push arbitrary progress events into the `"custom"` stream:

```python
from langgraph.config import get_stream_writer

def long_task(state: State) -> dict:
    writer = get_stream_writer()          # available inside any node
    writer({"progress": "fetching documents"})
    # ... work ...
    writer({"progress": "summarizing"})
    return {"messages": [AIMessage("done")]}

for chunk in graph.stream(inputs, config, stream_mode="custom"):
    print(chunk)   # {'progress': 'fetching documents'} ...
```

> **⚠️ Gotcha:** In **async** code on **Python < 3.11**, `get_stream_writer()` cannot find the run context. Instead, declare a `writer` parameter on the async node signature — LangGraph injects it: `async def node(state, writer): writer({...})`.

### Combining modes

Pass a list. Each chunk becomes a `(mode, data)` tuple so you can tell them apart:

```python
for mode, data in graph.stream(inputs, config, stream_mode=["updates", "custom"]):
    if mode == "updates":
        print("STEP:", data)
    elif mode == "custom":
        print("PROGRESS:", data)
```

### `astream_events` — the fine-grained event firehose

For framework-level introspection (every model start/stream/end, tool start/end, retriever events) across nested chains and subgraphs, use `astream_events` (v2 schema):

```python
async for event in graph.astream_events(inputs, config, version="v2"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
    elif kind == "on_tool_start":
        print(f"\n[tool start] {event['name']} args={event['data'].get('input')}")
```

> **✅ Best practice:** For most chat UIs, `stream_mode="messages"` is enough and far simpler. Reach for `astream_events` when you must observe *nested* component lifecycles (tools, retrievers, sub-chains) in one unified stream — e.g. to drive a rich activity timeline.

> **⚠️ Verify:** LangGraph is iterating on streaming. Recent versions added a `"checkpoints"`/`"tasks"` split and a typed event-projection API. The modes table above (`values`/`updates`/`messages`/`custom`/`debug`) and `astream_events(version="v2")` are stable and widely used; confirm any newer typed API against [the streaming docs](https://docs.langchain.com/oss/python/langgraph/streaming) for your installed version.

---

## 11. Durable execution, errors & runtime config

### Recursion limit

To prevent runaway loops (e.g. a tools↔agent cycle that never converges), LangGraph caps the number of super-steps. Exceeding it raises `GraphRecursionError`.

```python
from langgraph.errors import GraphRecursionError

try:
    graph.invoke(inputs, {"configurable": {"thread_id": "1"}, "recursion_limit": 10})
except GraphRecursionError:
    print("Loop did not converge within 10 steps — check your router/termination logic.")
```

> **Note:** `recursion_limit` counts **super-steps**, not LLM calls. Default is 25. A single agent turn (agent → tools → agent) is multiple steps, so size it with headroom but small enough to fail fast on infinite loops.

### Durable execution & recovery

With a persistent checkpointer, a graph that crashes mid-run can be **resumed from its last checkpoint** — just invoke again with the same `thread_id` and `None` input. Steps already completed are not re-run. This is what makes long-running and human-paused workflows production-safe. Make node functions **idempotent** where they perform external side effects, since a node can re-execute after a crash or interrupt.

### Passing runtime config/context to nodes

Nodes can accept a second parameter to read per-run configuration (user id, db handles, feature flags) passed under `configurable`:

```python
from langchain_core.runnables import RunnableConfig

def personalize(state: State, config: RunnableConfig) -> dict:
    user_id = config["configurable"].get("user_id", "anonymous")
    return {"messages": [AIMessage(f"Hello, {user_id}")]}

graph.invoke(
    {"messages": [HumanMessage("hi")]},
    {"configurable": {"thread_id": "t1", "user_id": "sam"}},
)
```

> **✅ Best practice:** Put **per-request, non-state** values (user id, tenant, request-scoped clients) in `configurable`, not in graph state. State is for data the graph *operates on and persists*; config is for the *environment of a single run*.

---

## 12. Complete worked example: plan → act loop with tools, checkpointing, and an approval interrupt

This ties the module together: a typed state, an agent↔tools cycle, SQLite persistence, and a human approval gate before any tool actually executes.

```python
from typing import Annotated, TypedDict, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command


# --- Tools -----------------------------------------------------------------
@tool
def create_invoice(customer: str, amount: float) -> str:
    """Create an invoice for a customer (a real side effect we want a human to approve)."""
    return f"Invoice created: {customer} owes ${amount:.2f}."


tools = [create_invoice]
tools_by_name = {t.name: t for t in tools}
llm = ChatAnthropic(model="claude-sonnet-4-6").bind_tools(tools)


# --- State -----------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# --- Nodes -----------------------------------------------------------------
def agent(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def human_gate(state: State) -> dict:
    """Pause for approval of EACH requested tool call before executing it."""
    last = state["messages"][-1]
    approvals = interrupt([
        {"name": tc["name"], "args": tc["args"]} for tc in last.tool_calls
    ])
    # `approvals` is the resumed value: a list of booleans, one per tool call.
    results = []
    for tc, ok in zip(last.tool_calls, approvals):
        if ok:
            output = tools_by_name[tc["name"]].invoke(tc["args"])
        else:
            output = "DENIED by human reviewer."
        results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
    return {"messages": results}


def route(state: State) -> Literal["human_gate", "__end__"]:
    last = state["messages"][-1]
    return "human_gate" if getattr(last, "tool_calls", None) else END


# --- Build -----------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("human_gate", human_gate)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route, {"human_gate": "human_gate", END: END})
builder.add_edge("human_gate", "agent")   # loop back so the model can react to results

with SqliteSaver.from_conn_string("plan_act.sqlite") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "session-1"}}

    # 1) Run until it pauses for approval.
    state = graph.invoke(
        {"messages": [HumanMessage("Invoice Acme Corp for $4,200.")]},
        config,
    )
    print("PENDING APPROVAL:", state["__interrupt__"])

    # 2) Human approves -> resume. (Send [False] to deny instead.)
    final = graph.invoke(Command(resume=[True]), config)
    print(final["messages"][-1].content)
    # -> The model confirms the invoice was created.
```

```mermaid
flowchart TD
    START((START)) --> A[agent]
    A -- tool_calls? --> H[human_gate]
    A -- no --> END((END))
    H -. interrupt: approve? .-> H
    H --> A
```

Notice the design choices: the tool's real side effect happens **after** the interrupt (in `human_gate`, post-`interrupt()`), so the "re-run from top" behavior never double-creates an invoice; state is minimal (just `messages`); the router is pure; and SQLite makes the pause survive a process restart.

---

## 13. Best practices

> **✅ Best practice — production checklist**
> - **Keep state minimal and typed.** Every channel is persisted on every step. Don't stash large blobs (raw documents, big payloads) in state; store a reference/id and rehydrate in the node.
> - **Choose reducers deliberately.** The reducer is your concurrency contract. Use `add_messages` for chat, `operator.add` for parallel list aggregation, default-overwrite for scalars.
> - **Name nodes clearly.** Node names appear in traces, streams, diagrams, and breakpoints. `validate_input` beats `node2`.
> - **Keep node functions pure and testable.** A node should be `state -> partial update`. Pull I/O behind injected clients (via `configurable`) so you can unit-test nodes without a live model.
> - **Put side effects after interrupts** (or make nodes idempotent), because interrupted/crashed nodes re-run from the top.
> - **Persist in production.** Use `PostgresSaver` (not `MemorySaver`) so memory, recovery, and human-in-the-loop survive restarts and scale across replicas.
> - **Set a `recursion_limit`** sized to fail fast on non-converging loops.
> - **Use `tools_condition` / `ToolNode`** for the standard agent loop instead of hand-rolling it — drop to manual nodes only when you need custom behavior (like the approval gate above).

---

## Recap

- LangGraph models computation as **nodes** (plain functions returning partial updates) over a **shared, typed State**, with **edges** controlling flow and **reducers** defining how concurrent writes merge.
- Unlike LCEL's acyclic pipe, LangGraph supports **cycles, persistent state, and human pauses** — the requirements of real agents.
- **State** is a `TypedDict` (or Pydantic/dataclass); `Annotated[list, add_messages]` and `Annotated[list, operator.add]` are the two reducers you'll reach for most; the default is overwrite. `MessagesState` is the chat shortcut.
- **`StateGraph`** → `add_node` / `add_edge` / `add_conditional_edges(source, router, mapping)` → `compile()` yields a `Runnable` `CompiledStateGraph`.
- **Conditional edges** build branches and the agent↔tools **cycle**; `tools_condition` is the prebuilt router.
- **`Send`** dynamically fans out N parallel node invocations (map-reduce), aggregated through a reducer.
- **Subgraphs** embed one compiled graph as a node — shared schema or via a transforming wrapper.
- **Checkpointers** (`MemorySaver` / `SqliteSaver` / `PostgresSaver`) snapshot state per super-step keyed by `thread_id`, enabling memory, durable recovery, `get_state` / `get_state_history` / `update_state`.
- **Human-in-the-loop**: dynamic `interrupt()` + `Command(resume=...)` (preferred), or static `interrupt_before` / `interrupt_after` breakpoints. Nodes re-run from the top on resume.
- **Time travel**: replay or fork from any historical checkpoint via `get_state_history` + a checkpoint config.
- **Streaming modes**: `values`, `updates`, `messages`, `custom` (stream writer), `debug` — combinable — plus `astream_events` for nested lifecycle events.
- Guard loops with `recursion_limit` (`GraphRecursionError`); pass per-run data via `RunnableConfig["configurable"]`.

---

## Exercises

1. **Reducer experiment.** Build a 3-channel `TypedDict`: one default-overwrite scalar, one `operator.add` list, and one custom reducer that keeps a running max. Write two nodes that both write all three keys, run sequentially, and confirm the merge behavior of each channel matches your expectation.

2. **Hand-built agent.** Without `create_react_agent`, build an agent↔tools loop using `ToolNode` and `tools_condition` for two tools of your choice (e.g. a calculator and a web-search stub). Add a `MemorySaver` and verify multi-turn memory across two `invoke` calls on the same `thread_id`.

3. **Map-reduce report.** Use the `Send` API to generate one paragraph per item in a runtime-supplied list of topics, then add a final "reduce" node that concatenates them into a single document. Confirm the workers run in parallel and that removing `operator.add` from the collection channel breaks it.

4. **Approval gate.** Take your agent from exercise 2 and insert a dynamic `interrupt()` that pauses before any tool executes, surfaces the pending tool call, and lets the caller approve or deny via `Command(resume=...)`. Ensure denying a call returns a `ToolMessage` so the model can react.

5. **Time travel.** Run a multi-step conversation with a checkpointer, walk `get_state_history`, pick a mid-conversation checkpoint, use `update_state` to change the user's last message, and continue from the fork. Compare the two timelines.

6. **(Stretch) Subgraph + streaming.** Extract your tool-execution logic into a subgraph with its own schema, embed it via a wrapper node, and stream the parent with `stream_mode=["updates", "custom"]` plus `subgraphs=True`, emitting a custom progress event from inside the subgraph.

---

**Previous:** [Module 8 — Agents with LangGraph](08-agents-with-langgraph.md) · **Next:** [Module 10 — Observability & Evaluation (LangSmith)](10-observability-and-eval-langsmith.md)

**Related:** [LCEL & the Runnable Interface](04-lcel-and-runnables.md) · [Tools & Tool Calling](05-tools-and-tool-calling.md) · [Memory & Conversation State](07-memory-and-state.md) · [Common Errors & Fixes](../appendix/B-common-errors.md) · [Glossary](../appendix/D-glossary.md) · [course home](../README.md)
