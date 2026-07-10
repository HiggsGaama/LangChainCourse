# Module 7 — Memory & Conversation State

Chat models are **pure functions**. Given a list of messages, they return a message. They remember nothing between calls. Every illusion of a model "remembering" your earlier turns is just your application re-sending the prior conversation on each request.

This module is about owning that illusion deliberately: how to store conversation history, how to feed it back, how to keep it from blowing past the context window, and — most importantly — the architectural shift from the LCEL-era memory helpers to **LangGraph persistence**, which is the modern foundation for anything stateful or agentic.

By the end you will be able to:

- Maintain multi-turn conversations manually and understand exactly what's happening on the wire.
- Use `RunnableWithMessageHistory` to auto-load/save history per session in an LCEL chain.
- Build a stateful chat app on **LangGraph + checkpointers** (the recommended modern path).
- Keep context windows under control with `trim_messages`, summarization, and `RemoveMessage`.
- Read and migrate away from the legacy `Conversation*Memory` classes you'll meet in old code.

---

## 1. The core problem: statelessness

Consider a naive two-turn exchange where we call the model twice with independent inputs:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-sonnet-4-6")

print(model.invoke("My name is Ada. Remember it.").content)
# "Got it, Ada! I'll remember your name for our conversation."

print(model.invoke("What is my name?").content)
# "I don't have access to your name. Could you tell me?"
```

The second call has **no idea** the first ever happened. The model's "I'll remember" was a lie it has no mechanism to keep. There is no server-side session; the LangChain `ChatModel` object holds no conversation state either.

The only way to get continuity is to **send the history back every time**:

```python
from langchain_core.messages import HumanMessage, AIMessage

history = [
    HumanMessage("My name is Ada. Remember it."),
]
reply = model.invoke(history)
history.append(reply)                       # persist the AI turn

history.append(HumanMessage("What is my name?"))
reply = model.invoke(history)
print(reply.content)
# "Your name is Ada."
```

That's the whole game. Everything in this module — message-history stores, `RunnableWithMessageHistory`, LangGraph checkpointers — is infrastructure for doing this **automatically, per user, durably, and without overflowing the context window**.

```mermaid
flowchart LR
    U[User turn] --> A[Append to history]
    A --> M[Model.invoke history]
    M --> R[AI reply]
    R --> A2[Append reply to history]
    A2 -. next turn .-> U
```

> **Note:** "Memory" in LangChain almost always means *the conversation transcript you replay*, not a learned long-term store. Long-term semantic memory (facts about a user that survive across threads) is a separate concern — typically built with a vector/SQL store, see [Retrieval & RAG](06-retrieval-and-rag.md) and the LangGraph store in [LangGraph Deep Dive](09-langgraph-deep-dive.md).

---

## 2. The manual baseline with `MessagesPlaceholder`

In real chains you rarely pass a bare list — you have a prompt with a system message and want history slotted in. That's what `MessagesPlaceholder` is for (see [Prompts](02-prompts.md)).

```python
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

model = init_chat_model("anthropic:claude-sonnet-4-6")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a terse assistant for a logistics company."),
    MessagesPlaceholder("history"),     # the running transcript drops in here
    ("human", "{input}"),
])

chain = prompt | model

history: list = []

def chat(user_text: str) -> str:
    response = chain.invoke({"history": history, "input": user_text})
    history.extend([HumanMessage(user_text), AIMessage(response.content)])
    return response.content

print(chat("Track shipment 4471."))
# "Shipment 4471 is in transit, ETA tomorrow 14:00."
print(chat("When did I ask about?"))
# "You asked about shipment 4471."
```

This works, but you own every concern manually: where `history` lives, how it's keyed per user, when it's persisted, and when it's trimmed. The rest of the module replaces this hand-rolled `history` variable with managed abstractions.

---

## 3. Chat message history stores

LangChain defines a storage interface, `BaseChatMessageHistory` (in `langchain_core.chat_history`), with the methods that matter:

- `messages` — property returning the stored `list[BaseMessage]`.
- `add_messages(messages)` / `add_message(message)` — append.
- `clear()` — wipe the thread.

Any backend that implements this interface is pluggable.

### In-memory

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

hist = InMemoryChatMessageHistory()
hist.add_messages([HumanMessage("Hi"), AIMessage("Hello!")])
print(hist.messages)
# [HumanMessage(content='Hi'), AIMessage(content='Hello!')]
```

`InMemoryChatMessageHistory` lives in the process heap — it vanishes on restart and isn't shared across workers. Fine for tests and notebooks, wrong for production.

### Persistent backends

These live in `langchain_community` (install the relevant driver):

```python
# SQL-backed (SQLite here; any SQLAlchemy URL works — Postgres, MySQL, ...)
# pip install langchain-community SQLAlchemy
from langchain_community.chat_message_histories import SQLChatMessageHistory

history = SQLChatMessageHistory(
    session_id="user-42",                       # rows are scoped by this id
    connection="sqlite:///conversations.db",    # use postgresql+psycopg://... in prod
)
history.add_user_message("What's my account balance?")
history.add_ai_message("Your balance is $1,204.50.")
print([m.content for m in history.messages])
```

```python
# Redis-backed
# pip install langchain-redis redis
from langchain_redis import RedisChatMessageHistory

history = RedisChatMessageHistory(
    session_id="user-42",
    redis_url="redis://localhost:6379",
    ttl=60 * 60 * 24,    # optional: expire the thread after 24h
)
```

> **⚠️ Gotcha:** The `connection` parameter on `SQLChatMessageHistory` superseded the older `connection_string` keyword. If you're on a recent `langchain-community`, use `connection`. Some integrations (e.g. older Redis history) moved out of `langchain_community` into dedicated packages like `langchain-redis` — check the import path against the version you installed.

> **✅ Best practice:** The `session_id` is your isolation boundary. Use a stable per-conversation key (e.g. `f"{user_id}:{conversation_id}"`), never a global constant — otherwise every user shares one transcript.

You *can* wire these stores into a chain by hand, but you'd be re-implementing load/append/save on every turn. The next abstraction does it for you.

---

## 4. `RunnableWithMessageHistory` — LCEL-era memory

`RunnableWithMessageHistory` (in `langchain_core.runnables.history`) wraps **any** runnable so that, per call, it:

1. Looks up the right history object via a `get_session_history` factory (keyed by `session_id`).
2. Loads stored messages and injects them into your `MessagesPlaceholder`.
3. Runs the underlying chain.
4. Appends both the new input and the output back into the store.

This is the **current LCEL-era memory primitive** — the supported replacement for the legacy `ConversationChain` + `Memory` classes (Section 7). For new *stateful/agentic* systems, prefer LangGraph (Section 5); but for a plain conversational chain, this is idiomatic and lightweight.

### Full runnable example

```python
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

model = init_chat_model("anthropic:claude-sonnet-4-6")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = prompt | model

# A process-local registry of session_id -> history object.
# In production, return a SQL/Redis-backed history here instead.
_store: dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]

conversational = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",      # which input field is the new human turn
    history_messages_key="history",  # which MessagesPlaceholder receives stored msgs
)
```

Now invoke it with a `configurable.session_id` so the wrapper knows *which* thread:

```python
cfg = {"configurable": {"session_id": "alice"}}

print(conversational.invoke({"input": "I'm planning a trip to Kyoto."}, config=cfg).content)
# "Nice! Kyoto is beautiful in autumn. What would you like to plan?"

print(conversational.invoke({"input": "What city did I mention?"}, config=cfg).content)
# "You mentioned Kyoto."

# A different session_id is a completely separate conversation:
print(conversational.invoke(
    {"input": "What city did I mention?"},
    config={"configurable": {"session_id": "bob"}},
).content)
# "You haven't mentioned a city yet."
```

```mermaid
flowchart TD
    I[invoke input + config session_id] --> F[get_session_history session_id]
    F --> L[load stored messages]
    L --> P[inject into history_messages_key]
    P --> C[run wrapped chain]
    C --> O[model output]
    O --> S[append input + output to store]
    S --> R[return output]
```

### The three keys, precisely

| Parameter | Meaning | When you need it |
|---|---|---|
| `input_messages_key` | Which field of the input dict holds the **new** human turn. | When your input is a dict (almost always). |
| `history_messages_key` | The `MessagesPlaceholder` variable that **stored** messages flow into. | When the prompt has a placeholder; without it, history is prepended to the input list. |
| `output_messages_key` | Which field of the output to persist as the AI turn. | Only when the wrapped runnable returns a **dict** (e.g. a chain ending in a parser). For a chain ending in a chat model — which returns a single `AIMessage` — leave it unset. |

`output_messages_key` example — a chain that returns a dict:

```python
chain_returning_dict = prompt | model | {"answer": lambda m: m.content}
wrapped = RunnableWithMessageHistory(
    chain_returning_dict,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",   # persist the "answer" field as the AI message
)
```

### Custom session keys

You can require more than `session_id` (e.g. `user_id` + `conversation_id`) by declaring `history_factory_config`:

```python
from langchain_core.runnables import ConfigurableFieldSpec

def get_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    return _store.setdefault((user_id, conversation_id), InMemoryChatMessageHistory())

conversational = RunnableWithMessageHistory(
    chain, get_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(id="user_id", annotation=str, is_shared=True),
        ConfigurableFieldSpec(id="conversation_id", annotation=str, is_shared=True),
    ],
)
conversational.invoke(
    {"input": "Hi"},
    config={"configurable": {"user_id": "u1", "conversation_id": "c1"}},
)
```

> **⚠️ Gotcha:** Streaming works (`.stream` / `.astream`), but the history is only written **after** the full output is produced. If a stream errors midway, nothing is persisted for that turn.

> **Note:** `RunnableWithMessageHistory` only stores *messages*. The moment you need to carry richer state — tool-call scratchpads, retrieved documents, a running summary, counters — you've outgrown it. That's the cue to move to LangGraph.

---

## 5. The strategic shift: LangGraph persistence

This is the most important section in the module.

`RunnableWithMessageHistory` is a thin layer that does one thing: replay a message list. Modern stateful applications — agents, multi-step workflows, anything with tools or branching — need to persist **arbitrary state**, resume after interruption, support time-travel/replay, and run as durable long-lived threads.

LangGraph provides this through **checkpointers**: after every step ("super-step") of a graph, the entire state is snapshotted to a checkpoint, keyed by a `thread_id`. Re-invoking with the same `thread_id` resumes from the last snapshot. The official guidance is that **for stateful/agentic systems, LangGraph persistence is the recommended way to do memory** — not the LCEL history wrapper.

### Minimal stateful chat app

```python
# pip install langgraph langchain-anthropic
from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

model = init_chat_model("anthropic:claude-sonnet-4-6")

class State(TypedDict):
    # add_messages is a *reducer*: new messages are APPENDED, not overwritten,
    # and updates are merged by message id.
    messages: Annotated[list, add_messages]

def call_model(state: State) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}     # appended via the reducer

builder = StateGraph(State)
builder.add_node("model", call_model)
builder.add_edge(START, "model")

# The checkpointer is what turns a stateless graph into a durable, resumable one.
checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)
```

Now memory is automatic — you only pass the **new** message each turn, and the checkpointer reloads the rest by `thread_id`:

```python
from langchain_core.messages import HumanMessage

cfg = {"configurable": {"thread_id": "session-1"}}

out = app.invoke({"messages": [HumanMessage("My name is Ada.")]}, config=cfg)
print(out["messages"][-1].content)
# "Nice to meet you, Ada!"

out = app.invoke({"messages": [HumanMessage("What's my name?")]}, config=cfg)
print(out["messages"][-1].content)
# "Your name is Ada."

# Different thread_id => clean slate:
out = app.invoke(
    {"messages": [HumanMessage("What's my name?")]},
    config={"configurable": {"thread_id": "session-2"}},
)
print(out["messages"][-1].content)
# "I don't know your name yet — what is it?"
```

You can inspect the persisted state at any time:

```python
snapshot = app.get_state(cfg)
print(len(snapshot.values["messages"]))   # 4  (2 human + 2 ai for session-1)
```

```mermaid
flowchart LR
    subgraph Thread[thread_id session-1]
      direction TB
      C0[checkpoint 0] --> C1[checkpoint 1] --> C2[checkpoint 2]
    end
    INV[app.invoke new msg] -->|load last checkpoint| C2
    C2 -->|run node, snapshot| C3[checkpoint 3]
```

### Production checkpointers

`MemorySaver` is in-process and ephemeral — the LangGraph analogue of `InMemoryChatMessageHistory`. For production swap in a durable backend; the graph code is unchanged, only the checkpointer object differs:

```python
# pip install langgraph-checkpoint-postgres psycopg[binary]
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:pass@localhost:5432/langgraph?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()                 # creates tables on first run
    app = builder.compile(checkpointer=checkpointer)
    app.invoke({"messages": [HumanMessage("hi")]},
               config={"configurable": {"thread_id": "t1"}})
```

There are also `SqliteSaver` (`langgraph-checkpoint-sqlite`) and async variants (`AsyncPostgresSaver`, `AsyncSqliteSaver`).

> **✅ Best practice:** New stateful or agentic project? Start on LangGraph + a checkpointer from day one. Retrofitting persistence later is painful. Use `RunnableWithMessageHistory` only for simple, non-agentic conversational chains.

The full graph model — reducers, multiple nodes, conditional edges, interrupts, the long-term `Store` — is covered in [Agents with LangGraph](08-agents-with-langgraph.md) and [LangGraph Deep Dive](09-langgraph-deep-dive.md).

---

## 6. Managing the context window

History grows without bound; context windows and budgets do not. Three techniques, often combined.

### 6.1 `trim_messages`

`trim_messages` (in `langchain_core.messages`) prunes a message list to fit a budget. Key parameters:

- `max_tokens` — the budget (token count, or message count when `token_counter=len`).
- `strategy` — `"last"` (keep most recent, the usual choice) or `"first"`.
- `token_counter` — a callable, or a chat model whose `get_num_tokens_from_messages` is used. Pass `len` to count *messages* instead of tokens.
- `include_system` — keep the leading `SystemMessage` even while trimming the rest (almost always `True`).
- `allow_partial` — allow splitting a single message to squeeze under the budget.
- `start_on` / `end_on` — ensure the trimmed list begins/ends on a given type (e.g. `start_on="human"`), so you never feed the model a dangling `ToolMessage` or an `AIMessage` first.

**Trim by message count** (provider-agnostic, no token math):

```python
from langchain_core.messages import (
    trim_messages, SystemMessage, HumanMessage, AIMessage,
)

messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Hi, I'm Ada."),
    AIMessage("Hello Ada!"),
    HumanMessage("I like hiking."),
    AIMessage("Great, hiking is wonderful."),
    HumanMessage("What's my name?"),
]

trimmed = trim_messages(
    messages,
    max_tokens=4,            # keep at most 4 messages (token_counter=len)
    strategy="last",
    token_counter=len,
    include_system=True,     # always retain the system prompt
    start_on="human",        # don't start on a stray AI/tool message
)
for m in trimmed:
    print(type(m).__name__, "-", m.content)
# SystemMessage - You are a helpful assistant.
# HumanMessage - I like hiking.
# AIMessage - Great, hiking is wonderful.
# HumanMessage - What's my name?
```

**Trim by real tokens**, using the model itself as the counter:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-sonnet-4-6")

trimmed = trim_messages(
    messages,
    max_tokens=2000,
    strategy="last",
    token_counter=model,     # uses model.get_num_tokens_from_messages
    include_system=True,
    allow_partial=False,
    start_on="human",
)
```

`trim_messages` is itself a `Runnable`, so it composes inline in LCEL:

```python
chain = trim_messages(max_tokens=2000, strategy="last", token_counter=model,
                      include_system=True, start_on="human") | model
```

> **⚠️ Gotcha:** With `strategy="last"` and tool-calling, an `AIMessage` that requests a tool must stay paired with its `ToolMessage` reply, or the provider rejects the request. Use `start_on="human"` and avoid `allow_partial` with tool flows to keep pairs intact.

> **⚠️ Verify:** Anthropic token counting through `get_num_tokens_from_messages` can be approximate, especially with tool schemas. If you need exact counts, call the provider's token-counting endpoint; otherwise leave headroom under the limit.

### 6.2 Summarizing older turns

Trimming *discards* old context. To retain its gist, summarize the oldest turns into a single message and keep only the recent verbatim:

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-haiku-4-5")  # cheap/fast for summarizing

def summarize_history(messages, keep_last: int = 4):
    if len(messages) <= keep_last:
        return messages
    old, recent = messages[:-keep_last], messages[-keep_last:]
    transcript = "\n".join(f"{type(m).__name__}: {m.content}" for m in old)
    summary = model.invoke(
        [HumanMessage(f"Summarize this conversation in 2-3 sentences:\n{transcript}")]
    ).content
    return [SystemMessage(f"Summary of earlier conversation: {summary}"), *recent]
```

In LangGraph this becomes a node that rewrites `state["messages"]` once it grows past a threshold (see next).

### 6.3 Deleting messages in LangGraph with `RemoveMessage`

Because `add_messages` merges by id, you delete history by emitting a special `RemoveMessage` (from `langchain_core.messages`) carrying the id to drop:

```python
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES   # sentinel id

def prune(state: State) -> dict:
    msgs = state["messages"]
    if len(msgs) <= 6:
        return {}
    # Drop everything except the last 4 messages.
    to_remove = [RemoveMessage(id=m.id) for m in msgs[:-4]]
    return {"messages": to_remove}
```

To clear an entire thread's messages in one shot, emit a single `RemoveMessage(id=REMOVE_ALL_MESSAGES)`. Wire `prune` as a node after your model node so the persisted state stays bounded.

> **⚠️ Gotcha:** Every message must have an `id` for `RemoveMessage` to target it. Messages created by the `add_messages` reducer are auto-assigned ids; if you hand-build messages, set `id` yourself.

---

## 7. Legacy memory classes (read-old-code reference)

> **LEGACY:** The `langchain.memory.Conversation*Memory` classes below are **deprecated**. You'll meet them in tutorials and existing codebases, so know what they did — but do not build new systems on them. The replacements are `RunnableWithMessageHistory` (simple chains) and **LangGraph state + `trim_messages`/summarization** (everything else).

| Legacy class | What it did | Modern replacement |
|---|---|---|
| `ConversationBufferMemory` | Stored the full transcript and replayed all of it. | Plain history store + `MessagesPlaceholder`; LangGraph `add_messages`. |
| `ConversationBufferWindowMemory` | Kept only the last *k* turns. | `trim_messages(max_tokens=k, strategy="last", token_counter=len)`. |
| `ConversationTokenBufferMemory` | Kept the last *N* **tokens**. | `trim_messages(max_tokens=N, token_counter=model, strategy="last")`. |
| `ConversationSummaryMemory` | Replaced history with a running LLM summary. | A summarization node (Section 6.2) over LangGraph state. |
| `ConversationSummaryBufferMemory` | Recent turns verbatim + summary of the rest. | Combine 6.2 (summary) and 6.1 (trim) in one node. |
| `ConversationEntityMemory` | Tracked per-entity facts ("Ada → likes hiking"). | Structured fields in LangGraph state + a long-term `Store` for cross-thread facts. |

These classes coupled storage, trimming, and prompt-injection into one opaque object that only worked with the now-legacy `ConversationChain` / `LLMChain`. The modern stack separates those concerns — storage (history store / checkpointer), reduction (`trim_messages` / summarization node), and prompting (`MessagesPlaceholder`) — which is why it composes cleanly with LCEL and LangGraph.

See [Versioning & Migration](../appendix/C-versioning-and-migration.md) for migration notes and [Common Errors & Fixes](../appendix/B-common-errors.md) for deprecation-warning fixes.

---

## 8. Best practices

> **✅ Best practice — isolate per user/session.** Key every store and every checkpointer by a stable, per-conversation id (`session_id` / `thread_id`). Treat a leaked or shared key as a security bug: one user reading another's transcript.

> **✅ Best practice — reduce before the model, not after.** Trim or summarize on the way *into* the model (e.g. `trim_messages | model`). You still persist the full transcript; you just don't pay to send all of it every turn. This also protects against context-window overflow errors.

> **✅ Best practice — persist durably in production.** `InMemoryChatMessageHistory` and `MemorySaver` lose everything on restart and aren't shared across workers/replicas. Use SQL/Redis history or a Postgres/SQLite checkpointer for anything real.

> **✅ Best practice — store *structured state* in LangGraph, not just messages.** If your app tracks more than a transcript — a user profile, a plan, retrieved docs, tool results — model it as typed fields in your graph state. Stuffing structured data into the message list as text is lossy and token-expensive.

> **🔧 Try it:** Take any chain from [LCEL & Runnables](04-lcel-and-runnables.md), wrap it in `RunnableWithMessageHistory`, then re-implement the same behavior as a one-node LangGraph app with `MemorySaver`. Compare how each handles a second `session_id` / `thread_id`.

---

## Recap

- Chat models are **stateless**; multi-turn continuity means replaying the transcript on every call.
- `MessagesPlaceholder` is where stored history is injected into a prompt.
- `BaseChatMessageHistory` is the storage interface: `InMemoryChatMessageHistory` for dev, `SQLChatMessageHistory` / `RedisChatMessageHistory` for persistence — always scoped by `session_id`.
- `RunnableWithMessageHistory` is the **LCEL-era** memory primitive: a `get_session_history` factory plus `input_messages_key` / `history_messages_key` / (optionally) `output_messages_key`, invoked with a `configurable.session_id`.
- **For stateful/agentic apps, prefer LangGraph persistence**: a checkpointer (`MemorySaver` → `PostgresSaver`/`SqliteSaver`) snapshots arbitrary state per `thread_id`, giving durable, resumable threads — not just message lists.
- Control the context window with `trim_messages` (by tokens or message count, `strategy="last"`, `include_system=True`, `start_on="human"`), summarization of old turns, and `RemoveMessage` to delete from LangGraph state.
- The `Conversation*Memory` classes are **legacy**; know them for reading old code, build new systems on the modern stack.

## Exercises

1. **Manual baseline.** Build a CLI loop using a `ChatPromptTemplate` with a `MessagesPlaceholder` and a hand-maintained message list. Confirm the assistant recalls a fact from three turns earlier, then deliberately drop the placeholder and watch it forget.
2. **Per-session wrapper.** Wrap a chain in `RunnableWithMessageHistory` backed by `SQLChatMessageHistory` over SQLite. Run two `session_id`s, restart the process, and verify each conversation reloads from disk independently.
3. **LangGraph chat.** Reproduce Exercise 2 as a single-node LangGraph app with `MemorySaver`, then swap to `SqliteSaver`. Use `app.get_state` to print how many messages are stored for a given `thread_id`.
4. **Trim to budget.** Feed a 20-message transcript through `trim_messages` with `strategy="last"`, `include_system=True`, and `token_counter=model`. Verify the system message survives and the list starts on a `HumanMessage`. Then compose it as `trim_messages(...) | model`.
5. **Summarize + prune.** Add a LangGraph node that, once `state["messages"]` exceeds 8 messages, summarizes the oldest into a `SystemMessage` and emits `RemoveMessage`s for the originals. Confirm via `get_state` that the stored list stays bounded.
6. **Migration.** Find or write a snippet using `ConversationBufferWindowMemory(k=3)` and rewrite it with `trim_messages(max_tokens=3, token_counter=len, strategy="last")` inside a `RunnableWithMessageHistory` chain — preserving identical behavior.
