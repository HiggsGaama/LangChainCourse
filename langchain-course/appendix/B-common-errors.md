# Appendix B — Common Errors & Fixes

A field guide to the errors you will actually hit building with modern LangChain (v0.3+, the `langchain-core` / partner-package era). Each entry follows the same shape: **Symptom → Cause → Fix**, with a runnable snippet where one helps.

Skim the section headers, find the message that matches your traceback, and jump in. Every entry cross-links to the module where the underlying concept is taught in depth.

> **Note:** Read the *whole* traceback, bottom line first. LangChain wraps a lot of provider SDK calls, so the real cause (a 401, a Pydantic `ValidationError`, a `KeyError`) is usually a few frames below the top-level `langchain_core` frame.

---

## 1. Imports & Packages

### 1.1 `ImportError: cannot import name 'X' from 'langchain'`

**Symptom**
```text
ImportError: cannot import name 'ChatAnthropic' from 'langchain.chat_models'
# or
ModuleNotFoundError: No module named 'langchain.embeddings'
```

**Cause** — In v0.3 the monolithic `langchain` package was split. Core abstractions live in `langchain_core`; each provider lives in its own *partner package* (`langchain_anthropic`, `langchain_openai`, …); long-tail integrations live in `langchain_community`. Old tutorials import from `langchain.*`, which either no longer exists or emits a deprecation shim.

**Fix** — Install the right package and import from it directly.

```bash
pip install langchain-anthropic langchain-openai langchain-community
```

```python
# ❌ old / deprecated
# from langchain.chat_models import ChatAnthropic
# from langchain.embeddings import OpenAIEmbeddings

# ✅ modern
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

Rule of thumb for *where* a symbol lives:

| You want… | Import from |
| --- | --- |
| `ChatPromptTemplate`, `StrOutputParser`, `RunnableLambda`, message types, `tool` | `langchain_core` |
| A specific provider's chat model / embeddings | its partner package (`langchain_anthropic`, `langchain_openai`, `langchain_google_genai`, …) |
| A vector store, document loader, or niche integration | `langchain_community` (or its own partner package, e.g. `langchain_chroma`) |
| Agents / graphs / checkpointers | `langgraph` |

> **🔧 Try it:** If you don't know where a symbol moved, search the API reference at `python.langchain.com/api_reference` or run `python -c "import langchain_core; help('langchain_core')"`. See [Module 0 — Orientation](../modules/00-orientation-and-ecosystem.md) for the full package map and [Appendix C — Versioning & Migration](C-versioning-and-migration.md) for the v0.2→v0.3 moves.

---

### 1.2 `No module named 'langchain_anthropic'` even though `langchain` is installed

**Symptom** — `init_chat_model("anthropic:…")` or `from langchain_anthropic import ChatAnthropic` raises `ModuleNotFoundError`.

**Cause** — Installing `langchain` does **not** pull in partner packages. They are deliberately separate so you only ship the SDKs you use.

**Fix**
```bash
pip install langchain-anthropic     # provider package (note the hyphen)
# import name uses the underscore:
# from langchain_anthropic import ChatAnthropic
```

> **⚠️ Gotcha:** The *pip* name uses hyphens (`langchain-anthropic`); the *import* name uses underscores (`langchain_anthropic`). Same for `langchain-openai` / `langchain_openai`.

---

## 2. Model Initialization

### 2.1 `init_chat_model`: "Unable to infer model provider"

**Symptom**
```text
ValueError: Unable to infer model provider for model='claude-sonnet-4-6',
please specify model_provider directly.
```

**Cause** — You passed a bare model name with no provider prefix, and the name isn't on `init_chat_model`'s small auto-inference list. Inference only works for a handful of well-known prefixes (e.g. `gpt-…` → OpenAI). Claude model IDs are not auto-inferred.

**Fix** — Use the explicit `provider:model` string (preferred) or the `model_provider=` kwarg.

```python
from langchain.chat_models import init_chat_model

# ✅ provider-prefixed string
llm = init_chat_model("anthropic:claude-sonnet-4-6")

# ✅ equivalent, explicit kwarg
llm = init_chat_model("claude-sonnet-4-6", model_provider="anthropic")

# swap providers by changing one string:
llm_oai = init_chat_model("openai:gpt-4.1")
```

> **Note:** `init_chat_model` lives in `langchain.chat_models` (the `langchain` umbrella package), not `langchain_core`. It still requires the relevant partner package to be installed — the string `"anthropic:…"` needs `langchain-anthropic` present.

### 2.2 "Unsupported provider" / `ImportError` from `init_chat_model`

**Symptom** — `ValueError: Unsupported {model_provider=}` or an `ImportError` naming the partner package.

**Cause** — Provider name typo (`anthropics`, `claude`) or the partner package isn't installed.

**Fix** — Use a supported provider key (`anthropic`, `openai`, `google_genai`, `groq`, `mistralai`, `ollama`, …) and `pip install` its package. See [Module 1 — Models](../modules/01-models-chat-and-llms.md).

---

## 3. Authentication & API Keys

### 3.1 401 / `AuthenticationError` / "invalid x-api-key"

**Symptom**
```text
anthropic.AuthenticationError: Error code: 401 - invalid x-api-key
# or
openai.AuthenticationError: Error code: 401
```

**Cause** — Missing, misspelled, or wrong-provider env var. Each partner package reads a **specific** variable name.

**Fix** — Set the correct variable *before* the process starts (or load it before constructing the model).

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."   # Anthropic / ChatAnthropic
# os.environ["OPENAI_API_KEY"]   = "sk-..."       # OpenAI
# os.environ["GOOGLE_API_KEY"]   = "..."          # google_genai

from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-6")
```

| Provider | Env var |
| --- | --- |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |

> **✅ Best practice:** Keep keys in a `.env` file (gitignored) and load with `python-dotenv`'s `load_dotenv()` at the top of your entrypoint. You can also pass the key explicitly: `ChatAnthropic(model=..., api_key=os.environ["ANTHROPIC_API_KEY"])`.

> **⚠️ Gotcha:** In notebooks, setting `os.environ[...]` *after* you already constructed the model is fine (the key is read at request time for most clients), but setting it in a different kernel/terminal than the one running your code is the classic "I set it but it's still 401" trap. Verify with `print(bool(os.environ.get("ANTHROPIC_API_KEY")))`.

---

## 4. Pydantic v1 vs v2

### 4.1 `langchain_core.pydantic_v1` deprecation / mixing v1 and v2 models

**Symptom**
```text
LangChainDeprecationWarning: Importing from langchain_core.pydantic_v1 is deprecated.
# or, at runtime:
pydantic.errors.PydanticSchemaGenerationError / unexpected keyword arguments
```

**Cause** — Older code imported `BaseModel` from the `langchain_core.pydantic_v1` compatibility shim. Modern LangChain targets **Pydantic v2** directly. Mixing a v1-shim model with v2 internals causes schema-generation failures.

**Fix** — Import Pydantic v2 directly.

```python
# ❌ deprecated shim
# from langchain_core.pydantic_v1 import BaseModel, Field

# ✅ Pydantic v2
from pydantic import BaseModel, Field

class WeatherReport(BaseModel):
    city: str = Field(description="City name")
    temp_c: float = Field(description="Temperature in Celsius")
```

> **⚠️ Gotcha:** Don't pass a **v1** `BaseModel` subclass to `with_structured_output` in a v2 environment. Keep every schema on `pydantic` (v2). See [Appendix C](C-versioning-and-migration.md).

### 4.2 `with_structured_output` raises `ValidationError: field required`

**Symptom**
```text
pydantic_core._pydantic_core.ValidationError: 1 validation error for WeatherReport
temp_c
  Field required [type=missing, ...]
```

**Cause** — The model returned JSON missing a required field (the schema asked for something the prompt never elicited), or your field descriptions are too vague for the model to populate them reliably.

**Fix** — (a) Make truly-optional fields optional; (b) write strong `Field(description=...)` text; (c) inspect what actually came back with `include_raw=True` (see §6.1).

```python
from typing import Optional
from pydantic import BaseModel, Field

class WeatherReport(BaseModel):
    city: str = Field(description="City name the user asked about")
    temp_c: Optional[float] = Field(
        default=None, description="Temperature in Celsius, if stated"
    )
```

See [Module 3 — Output Parsers & Structured Output](../modules/03-output-parsers-structured-output.md).

---

## 5. Tools & Tool Calling

### 5.1 "Model does not support tools" / tools silently ignored

**Symptom** — `bind_tools` raises, or the model just answers in prose and never emits `tool_calls`.

**Cause** — The chosen model/endpoint doesn't support tool calling (e.g. a legacy completion `LLM` rather than a chat model, or a tiny local model without the capability).

**Fix** — Use a tool-calling chat model (all current Claude models support tools). Verify by inspecting the response.

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

llm = ChatAnthropic(model="claude-sonnet-4-6").bind_tools([add])
msg = llm.invoke("What is 21 + 21?")
print(msg.tool_calls)
# [{'name': 'add', 'args': {'a': 21, 'b': 21}, 'id': 'toolu_...', 'type': 'tool_call'}]
```

### 5.2 Malformed `args_schema` / bad tool signature

**Symptom** — `tool_calls` come back with wrong or missing args, or `bind_tools` complains about the schema.

**Cause** — The tool has no docstring (the model has nothing to infer purpose from), untyped parameters, or a hand-written `args_schema` that doesn't match the function.

**Fix** — Type every parameter, write a real docstring, and describe args. The `@tool` decorator derives the schema from the signature + docstring.

```python
from typing import Literal
from langchain_core.tools import tool

@tool
def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. "Paris".
        unit: Temperature unit, "c" for Celsius or "f" for Fahrenheit.
    """
    return f"20 {unit.upper()} in {city}"
```

See [Module 5 — Tools & Tool Calling](../modules/05-tools-and-tool-calling.md).

### 5.3 `BadRequestError`: tool_calls without a following ToolMessage

**Symptom**
```text
anthropic.BadRequestError: messages: tool_use ids were found without
tool_result blocks immediately after
# (OpenAI variant: "An assistant message with 'tool_calls' must be followed by
#  tool messages responding to each 'tool_call_id'")
```

**Cause** — You sent an `AIMessage` that contains `tool_calls` back to the model **without** appending one `ToolMessage` per tool call. The provider requires every `tool_call` to be answered before the next turn.

**Fix** — Execute each tool call and append a `ToolMessage` whose `tool_call_id` matches the call's `id`.

```python
from langchain_core.messages import HumanMessage, ToolMessage

tools_by_name = {"add": add}
messages = [HumanMessage("What is 21 + 21?")]
ai_msg = llm.invoke(messages)
messages.append(ai_msg)

for call in ai_msg.tool_calls:
    result = tools_by_name[call["name"]].invoke(call["args"])
    messages.append(
        ToolMessage(content=str(result), tool_call_id=call["id"])  # id MUST match
    )

final = llm.invoke(messages)   # now valid
```

> **✅ Best practice:** Don't hand-roll this loop in production. LangGraph's prebuilt `ToolNode` / `create_react_agent` wires `tool_call_id` matching for you — see §8 and [Module 8 — Agents with LangGraph](../modules/08-agents-with-langgraph.md).

### 5.4 `tool_call_id` mismatch

**Symptom** — Same `BadRequestError` family, or "unexpected tool_result".

**Cause** — You generated a fresh id, reordered messages, or dropped the originating `AIMessage`. The `ToolMessage.tool_call_id` must equal the `id` from the `AIMessage.tool_calls` that requested it, and the `AIMessage` must remain in history.

**Fix** — Always copy the id from the call (`call["id"]`), keep the `AIMessage` immediately before its `ToolMessage`s, and answer **every** call (no partial responses).

---

## 6. Structured Output Edge Cases

### 6.1 `with_structured_output` returns `None` or raises validation

**Symptom** — `.invoke(...)` returns `None`, or you get a `ValidationError` you can't explain.

**Cause** — The model produced text/partial JSON that couldn't be parsed into the schema. By default you only see the parsed object (or the failure), not what the model actually said.

**Fix** — Pass `include_raw=True` to get a dict of `{"raw", "parsed", "parsing_error"}` so you can inspect the underlying message.

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

structured = llm.with_structured_output(Person, include_raw=True)
out = structured.invoke("Ada Lovelace, age 36")
print(out["parsed"])         # Person(name='Ada Lovelace', age=36) or None
print(out["parsing_error"])  # None, or the exception
print(out["raw"])            # the raw AIMessage — see what the model emitted
```

### 6.2 Provider can't honor the structured-output `method`

**Symptom** — Empty/garbage output, or "function calling not supported" for that method.

**Cause** — `with_structured_output` supports different `method`s (`"function_calling"`, `"json_mode"`, `"json_schema"`). Not every provider supports every method, and the default differs by provider.

**Fix** — Try an alternate method explicitly.

```python
# tool/function-calling based (broadly supported)
structured = llm.with_structured_output(Person, method="function_calling")

# native JSON mode, where supported
# structured = llm.with_structured_output(Person, method="json_mode")
```

> **🔧 Try it:** If one method yields `None`, switch methods and re-run with `include_raw=True`. More in [Module 3](../modules/03-output-parsers-structured-output.md).

---

## 7. LCEL, Runnables & Streaming

### 7.1 `.stream()` yields nothing, or everything at once at the end

**Symptom** — You call `chain.stream(...)` but tokens arrive in one big lump, or only after the whole chain finishes.

**Cause** — A **non-streaming step buffers the chain**. Streaming is only end-to-end if every step can stream incrementally. The usual culprits:
1. A step that must see the *entire* upstream output before producing anything — e.g. `PydanticOutputParser`, `JsonOutputParser` over a full object, or a custom `RunnableLambda` that consumes the whole input.
2. You called `.invoke()` instead of `.stream()` (or iterated the result wrong).
3. A custom function in the chain returns a value rather than yielding.

**Fix** — Stream from the model directly when you need token-level output; keep buffering parsers *after* the point you stream, or use streaming-aware parsers.

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Write one sentence about {topic}.")
chain = prompt | ChatAnthropic(model="claude-sonnet-4-6") | StrOutputParser()

for chunk in chain.stream({"topic": "otters"}):   # StrOutputParser streams fine
    print(chunk, end="", flush=True)
```

> **Note:** `StrOutputParser` is streaming-friendly (it passes chunks through). `JsonOutputParser` can emit *partial* JSON objects as they grow. A Pydantic parser that validates a complete object cannot stream — it will buffer. See [Module 4 — LCEL & the Runnable Interface](../modules/04-lcel-and-runnables.md).

> **🔧 Try it:** For event-level visibility (which step is streaming, tool starts, etc.) use `async for event in chain.astream_events(input, version="v2")`.

### 7.2 Dict / function coercion surprises in LCEL

**Symptom** — `TypeError`, or a step receives a dict when it expected a string (or vice versa).

**Cause** — LCEL auto-coerces: a plain `dict` in a pipe becomes a `RunnableParallel` (each value run on the same input); a plain `callable` becomes a `RunnableLambda`. If your function signature doesn't match what flows in, it breaks.

**Fix** — Make the data shape explicit. Use `RunnablePassthrough.assign(...)` to add keys, and wrap functions deliberately.

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# A dict literal => RunnableParallel: both keys get the SAME input
chain = {
    "context": RunnableLambda(lambda x: retrieve(x["question"])),
    "question": RunnablePassthrough(),
} | prompt | llm
```

### 7.3 Lost callbacks / no tracing through a custom function

**Symptom** — Your LangSmith trace shows a gap; spans inside a custom step are missing; callbacks don't fire downstream of your function.

**Cause** — A custom function invoked a nested runnable **without forwarding `RunnableConfig`**. Callbacks, tracing, tags, and metadata propagate via `config` — drop it and the child run is orphaned.

**Fix** — Accept `config: RunnableConfig` and pass it through to every nested `.invoke()` / `.stream()`.

```python
from langchain_core.runnables import RunnableConfig, RunnableLambda

def my_step(x: dict, config: RunnableConfig) -> str:
    # forward config so the nested call stays in the same trace
    return summarizer.invoke(x["text"], config=config)

step = RunnableLambda(my_step)
```

> **✅ Best practice:** Any custom `RunnableLambda` that calls another runnable should take `config` as its second positional arg and forward it. See [Module 4](../modules/04-lcel-and-runnables.md) and [Module 10 — Observability & Evaluation](../modules/10-observability-and-eval-langsmith.md).

---

## 8. LangGraph

### 8.1 `GraphRecursionError`

**Symptom**
```text
langgraph.errors.GraphRecursionError: Recursion limit of 25 reached without
hitting a stop condition.
```

**Cause** — The graph keeps looping (e.g. an agent that calls a tool, gets a result, calls again forever) and never routes to `END`. The default `recursion_limit` is 25 *super-steps*.

**Fix** — Fix the routing logic so a stop condition is reachable; raise the limit only if the workload legitimately needs more steps.

```python
result = graph.invoke(
    {"messages": [("user", "…")]},
    config={"recursion_limit": 50},   # band-aid; first check your edges/conditions
)
```

> **⚠️ Gotcha:** A high recursion limit hides a real bug most of the time. Inspect the conditional edge that should route to `END`. See [Module 9 — LangGraph Deep Dive](../modules/09-langgraph-deep-dive.md).

### 8.2 "node 'X' not found" / edge references unknown node

**Symptom**
```text
ValueError: Found edge ending at unknown node 'tools'
```

**Cause** — You added an edge to/from a node name you never registered with `add_node`, or a typo between the node name and the edge target.

**Fix** — Names are strings and must match exactly. Register the node before referencing it in an edge.

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)        # must exist before the edge
builder.add_edge(START, "agent")
builder.add_edge("tools", "agent")
graph = builder.compile()
```

### 8.3 `add_messages` not used → messages overwritten instead of appended

**Symptom** — Conversation history "resets" each step; the state's `messages` only ever holds the latest update.

**Cause** — Each node returns a partial state update; LangGraph **merges** updates using a *reducer*. Without a reducer, the default behavior is to **overwrite** the key. For a running message list you must annotate the field with the `add_messages` reducer.

**Fix**
```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]   # appends + dedupes by id
```

### 8.4 `InvalidUpdateError`: concurrent updates to one key without a reducer

**Symptom**
```text
langgraph.errors.InvalidUpdateError: At key 'X': Can receive only one value per step.
Use an Annotated key to handle multiple values.
```

**Cause** — Two branches that run in the same super-step both wrote the same state key, but that key has no reducer to combine the writes.

**Fix** — Give the key a reducer that merges concurrent values (e.g. `operator.add` to concatenate lists, or `add_messages`).

```python
import operator
from typing import Annotated, TypedDict

class State(TypedDict):
    results: Annotated[list, operator.add]   # parallel branches' lists concatenate
```

### 8.5 Checkpointer used without a `thread_id`

**Symptom**
```text
ValueError: Checkpointer requires one or more of the following 'configurable'
keys: ['thread_id'] ...
```

**Cause** — You compiled the graph with a checkpointer (for persistence/memory) but invoked it without `configurable.thread_id`. The checkpointer keys state by thread.

**Fix** — Pass a `thread_id` in `config`.

```python
from langgraph.checkpoint.memory import InMemorySaver

graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "user-42"}}
graph.invoke({"messages": [("user", "hi")]}, config=config)
graph.invoke({"messages": [("user", "what did I just say?")]}, config=config)  # remembers
```

> **Note:** `InMemorySaver` is for dev only — state is lost on process exit. For production use `langgraph-checkpoint-sqlite` or `langgraph-checkpoint-postgres`. See [Module 7 — Memory & State](../modules/07-memory-and-state.md) and [Module 11 — Production & Deployment](../modules/11-production-and-deployment.md).

---

## 9. Context Length

### 9.1 "context length exceeded" / `max_tokens`-style error

**Symptom**
```text
anthropic.BadRequestError: ... prompt is too long: N tokens > limit
# OpenAI: "This model's maximum context length is N tokens..."
```

**Cause** — Conversation history, retrieved chunks, or a giant document pushed the prompt past the model's context window.

**Fix** — Reduce what you send: trim or summarize history, and use smaller retrieval chunks.

```python
from langchain_core.messages import trim_messages, SystemMessage, HumanMessage

trimmer = trim_messages(
    max_tokens=4000,
    strategy="last",            # keep the most recent messages
    token_counter=llm,          # use the model's tokenizer
    include_system=True,        # always keep the system prompt
    start_on="human",           # ensure history starts on a human turn
)
trimmed = trimmer.invoke(long_message_history)
```

Other levers:
- **Summarize** older turns into a single running summary message (see [Module 7](../modules/07-memory-and-state.md)).
- **Smaller chunks / higher overlap tradeoff** and tighter top-`k` in retrieval (see [Module 6 — Retrieval & RAG](../modules/06-retrieval-and-rag.md)).

---

## 10. Rate Limits & Resilience

### 10.1 HTTP 429 / `RateLimitError`

**Symptom**
```text
anthropic.RateLimitError: Error code: 429
```

**Cause** — Too many requests/tokens per minute for your tier, or bursty parallel calls.

**Fix** — Combine three independent tools:

**(a) Retry with backoff** for transient 429s:
```python
llm_resilient = llm.with_retry(
    retry_if_exception_type=(Exception,),
    stop_after_attempt=4,        # exponential backoff between attempts
)
```

**(b) Proactively throttle** with a client-side rate limiter (smooths bursts so you don't 429 in the first place):
```python
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_anthropic import ChatAnthropic

limiter = InMemoryRateLimiter(
    requests_per_second=2,       # token-bucket refill rate
    check_every_n_seconds=0.1,
    max_bucket_size=5,           # allow short bursts
)
llm = ChatAnthropic(model="claude-sonnet-4-6", rate_limiter=limiter)
```

**(c) Fall back** to another model when the primary is exhausted:
```python
primary = ChatAnthropic(model="claude-sonnet-4-6")
backup = ChatAnthropic(model="claude-haiku-4-5")
llm = primary.with_fallbacks([backup])
```

> **Note:** `InMemoryRateLimiter` limits *request rate*, not tokens, and is per-process (it won't coordinate across workers). For multi-process deployments enforce limits at a gateway/proxy. See [Module 1](../modules/01-models-chat-and-llms.md) and [Module 11](../modules/11-production-and-deployment.md).

---

## 11. Async in Jupyter / Event Loops

### 11.1 `RuntimeError: This event loop is already running` / `asyncio.run() cannot be called from a running event loop`

**Symptom** — Calling `asyncio.run(chain.ainvoke(...))` inside Jupyter blows up.

**Cause** — Jupyter (IPython) already runs an event loop in the kernel. You can't start a *second* one with `asyncio.run()`.

**Fix** — In a notebook cell, just `await` the coroutine directly (cells are async-aware):

```python
result = await chain.ainvoke({"topic": "otters"})   # ✅ in Jupyter, no asyncio.run
```

If you're stuck in a context that genuinely needs a nested loop (e.g. a library that calls `asyncio.run` internally), apply `nest_asyncio` as a last resort:

```python
import nest_asyncio
nest_asyncio.apply()
# now asyncio.run(...) works inside the existing loop
```

> **⚠️ Gotcha:** `nest_asyncio` is a workaround, not a fix — prefer `await` in notebooks and proper async entrypoints in apps. See [Module 4](../modules/04-lcel-and-runnables.md).

---

## 12. Caching & Tracing Not Working

### 12.1 LLM cache has no effect

**Symptom** — Identical calls still hit the API; no speedup.

**Cause** — `set_llm_cache(...)` was never called, was called *after* the model was used, or you expected caching across processes from an in-memory cache.

**Fix** — Set the global cache **once, early**, before invoking models.

```python
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

set_llm_cache(InMemoryCache())   # do this at startup, before any .invoke()
```

> **Note:** `InMemoryCache` is per-process and lost on restart. For a persistent/shared cache use a SQLite or Redis cache backend. Caching keys on the exact prompt + params, so any change (temperature, model id) is a cache miss.

### 12.2 LangSmith traces not appearing

**Symptom** — Runs execute fine but nothing shows up in LangSmith.

**Cause** — Tracing env vars not set, set in the wrong process, the SDK not installed, or the program exited before background traces flushed.

**Fix** — Set the `LANGSMITH_*` env vars before importing/running, and install `langsmith`.

```bash
pip install langsmith
```
```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "lsv2_..."
os.environ["LANGSMITH_PROJECT"] = "my-project"   # optional; defaults to "default"
# (legacy names LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY still work)
```

> **⚠️ Gotcha:** Tracing is uploaded in the background. In short-lived scripts the process can exit before the queue flushes. Use the SDK's tracing context (`from langsmith import tracing_context`) or let the program live long enough to flush; long-running servers are unaffected. See [Module 10 — Observability & Evaluation](../modules/10-observability-and-eval-langsmith.md).

---

## Recap

- **Imports moved** in v0.3: core in `langchain_core`, providers in partner packages (`langchain_anthropic`, …), niche integrations in `langchain_community`. Pip names use hyphens, import names use underscores.
- **`init_chat_model`** needs a `provider:model` string (Claude IDs aren't auto-inferred) and the partner package installed.
- **Auth** errors are almost always the wrong env var name or the wrong process.
- **Pydantic v2** is the standard; drop the `langchain_core.pydantic_v1` shim and never mix versions in `with_structured_output`.
- **Tool calling** requires one `ToolMessage` per `tool_call`, with matching `tool_call_id`, after the `AIMessage` — let LangGraph's `ToolNode` handle it.
- **`include_raw=True`** is your debugger for `with_structured_output`.
- **Streaming** is only end-to-end if every step streams; buffering parsers and `.invoke()` defeat it.
- **Forward `RunnableConfig`** through custom functions or you lose tracing and callbacks.
- **LangGraph** errors usually trace to a missing reducer (`add_messages`), an unreachable `END` (recursion), a missing `thread_id`, or a node-name typo.
- **Context, rate limits, async, caching, tracing**: each has a one-line idiomatic fix — `trim_messages`, `with_retry` + `InMemoryRateLimiter` + `with_fallbacks`, `await` (or `nest_asyncio`), early `set_llm_cache`, and `LANGSMITH_*` env vars set before run.

> **🔧 Try it:** When stuck, reproduce the failure in the smallest possible script, add `include_raw=True` / `astream_events` / a LangSmith trace, and read the bottom of the traceback first. See also [Appendix A — Cheat Sheets](A-cheatsheets.md), [Appendix C — Versioning & Migration](C-versioning-and-migration.md), and [Appendix D — Glossary](D-glossary.md).
