# Appendix D — Glossary

A precise, alphabetical reference of the core LangChain and LangGraph terms used throughout this course. Each entry gives a one- or two-sentence definition and a link to the module where the concept is taught in depth. Terms are grouped by first letter for quick scanning.

> **Note:** Definitions target **modern LangChain (v0.3+)** — the `langchain-core` / partner-package era. Where a term names a specific class, the originating package is given in `code font`.

---

## A

**Agent**
An LLM-driven system that decides *which* actions to take (typically tool calls) and in *what order*, looping until it reaches a final answer. In modern LangChain, agents are built as graphs with `langgraph` rather than the legacy `AgentExecutor`. See [Module 8 — Agents with LangGraph](../modules/08-agents-with-langgraph.md).

**AIMessage**
The message type (`langchain_core.messages.AIMessage`) representing a model's response; it carries `content`, optional `tool_calls`, and `response_metadata` such as token usage. See [Module 1 — Models](../modules/01-models-chat-and-llms.md).

**astream_events**
An async method on every `Runnable` that emits a fine-grained stream of lifecycle events (`on_chat_model_stream`, `on_tool_end`, etc.) from every step in a chain or graph — the primary way to surface token-level and intermediate progress in UIs. See [Module 4 — LCEL & the Runnable Interface](../modules/04-lcel-and-runnables.md).

## B

**BaseChatModel**
The abstract base class (`langchain_core.language_models.BaseChatModel`) that all chat model integrations subclass; it defines the message-in/message-out contract (`invoke`, `stream`, `bind_tools`, `with_structured_output`). See [Module 1 — Models](../modules/01-models-chat-and-llms.md).

**Batch**
The `Runnable.batch` / `abatch` method that runs many inputs through a chain concurrently (with a configurable `max_concurrency`), more efficient than looping over `invoke`. See [Module 4 — LCEL & the Runnable Interface](../modules/04-lcel-and-runnables.md).

## C

**Callback / CallbackHandler**
The event-hook system (`langchain_core.callbacks.BaseCallbackHandler`) that fires on model starts, token streams, tool calls, errors, etc.; used for logging, streaming, and tracing (LangSmith is built on it). See [Module 10 — Observability & Evaluation](../modules/10-observability-and-eval-langsmith.md).

**Chain**
Any composition of components that transforms an input into an output; in modern LangChain, chains are `Runnable`s composed with the LCEL pipe (`|`) operator rather than legacy `Chain` subclasses. See [Module 4 — LCEL & the Runnable Interface](../modules/04-lcel-and-runnables.md).

**Channel (LangGraph)**
The low-level storage primitive behind a graph's state; each state key is backed by a channel whose **reducer** determines how concurrent writes are combined. See [Module 9 — LangGraph Deep Dive](../modules/09-langgraph-deep-dive.md).

**Checkpointer**
A LangGraph persistence backend (e.g. `langgraph.checkpoint.memory.InMemorySaver`, or SQLite/Postgres savers) that snapshots graph state after each step, enabling memory, resumption, time-travel, and human-in-the-loop. See [Module 9 — LangGraph Deep Dive](../modules/09-langgraph-deep-dive.md).

**Chat model**
A model that consumes a list of messages and returns an `AIMessage`; the standard interface in LangChain, typically constructed via `init_chat_model` or a class like `ChatAnthropic`. See [Module 1 — Models](../modules/01-models-chat-and-llms.md).

**Chunk / chunking**
A chunk is one segment of a larger document produced by a text splitter (e.g. `RecursiveCharacterTextSplitter`); chunking controls retrieval granularity in RAG. (The same word also denotes one streamed piece of a model response — an `AIMessageChunk`.) See [Module 6 — Retrieval & RAG](../modules/06-retrieval-and-rag.md).

## D

**Document**
The container type (`langchain_core.documents.Document`) holding `page_content` (text) plus a `metadata` dict; the unit that loaders produce, splitters chunk, and retrievers return. See [Module 6 — Retrieval & RAG](../modules/06-retrieval-and-rag.md).

## E

**Embeddings**
A model (interface `langchain_core.embeddings.Embeddings`) that maps text to dense numeric vectors so semantic similarity becomes a distance computation; the backbone of vector search in RAG. See [Module 6 — Retrieval & RAG](../modules/06-retrieval-and-rag.md).

## F

**Few-shot prompting**
A prompting technique that includes a handful of input/output examples in the prompt to steer behavior; supported by `FewShotChatMessagePromptTemplate` and example selectors. See [Module 2 — Prompts & Prompt Templates](../modules/02-prompts.md).

## H

**HumanMessage**
The message type (`langchain_core.messages.HumanMessage`) representing user input in a conversation. See [Module 1 — Models](../modules/01-models-chat-and-llms.md).

## I

**init_chat_model**
A provider-agnostic factory (`langchain.chat_models.init_chat_model`) that builds a chat model from a string like `"anthropic:claude-sonnet-4-6"`, letting you swap providers without changing class imports. See [Module 1 — Models](../modules/01-models-chat-and-llms.md).

## L

**LangGraph**
A library (`langgraph`) for building stateful, multi-step LLM applications as graphs of nodes and edges with built-in persistence; the modern foundation for agents and complex control flow. See [Module 8 — Agents with LangGraph](../modules/08-agents-with-langgraph.md) and [Module 9 — LangGraph Deep Dive](../modules/09-langgraph-deep-dive.md).

**LangServe**
A library that turns a `Runnable` into a production REST API (FastAPI-based) with auto-generated endpoints and a playground; note that for graph-based apps the **LangGraph Platform / Server** is now the preferred deployment path. See [Module 11 — Production & Deployment](../modules/11-production-and-deployment.md).

**LangSmith**
A framework-agnostic observability and evaluation platform from the LangChain team for tracing runs, debugging, and running evals/datasets; integrates automatically via environment variables. See [Module 10 — Observability & Evaluation](../modules/10-observability-and-eval-langsmith.md).

**LCEL (LangChain Expression Language)**
The declarative composition syntax built on the `Runnable` interface, using the `|` pipe operator to wire components into chains that automatically support `invoke`, `stream`, `batch`, and async. See [Module 4 — LCEL & the Runnable Interface](../modules/04-lcel-and-runnables.md).

## M

**MessagesPlaceholder**
A prompt-template slot (`langchain_core.prompts.MessagesPlaceholder`) that injects a variable-length list of messages (e.g. chat history) into a `ChatPromptTemplate`. See [Module 2 — Prompts & Prompt Templates](../modules/02-prompts.md).

**MMR (Maximal Marginal Relevance)**
A retrieval/search strategy that balances relevance against diversity, reducing redundant near-duplicate results; selectable via `search_type="mmr"` on many retrievers. See [Module 6 — Retrieval & RAG](../modules/06-retrieval-and-rag.md).

**Multimodal**
The ability of a model (e.g. Claude) to accept content beyond text — such as images or documents — passed as structured content blocks in a `HumanMessage`. See [Module 1 — Models](../modules/01-models-chat-and-llms.md).

## O

**Output parser**
A `Runnable` (`langchain_core.output_parsers`) that converts raw model output into a structured form, e.g. `StrOutputParser`, `JsonOutputParser`, or `PydanticOutputParser`. See [Module 3 — Output Parsers & Structured Output](../modules/03-output-parsers-structured-output.md).

## P

**Partner package**
A standalone, separately versioned integration package maintained alongside `langchain-core` (e.g. `langchain_anthropic`, `langchain_openai`); the recommended source for provider classes. See [Module 0 — Orientation & The LangChain Ecosystem](../modules/00-orientation-and-ecosystem.md).

**Prompt template**
A reusable, parameterized prompt object (`PromptTemplate` or `ChatPromptTemplate` in `langchain_core.prompts`) whose `.invoke()` fills in variables and returns a `PromptValue`. See [Module 2 — Prompts & Prompt Templates](../modules/02-prompts.md).

**PromptValue**
The provider-neutral output of a prompt template that can be rendered either to a string (`to_string()`) or to messages (`to_messages()`), letting the same prompt feed an LLM or a chat model. See [Module 2 — Prompts & Prompt Templates](../modules/02-prompts.md).

**Pydantic**
The data-validation library LangChain uses (v2) to define typed schemas for structured output and tool arguments; you pass a `BaseModel` subclass to `with_structured_output` or as a tool's `args_schema`. See [Module 3 — Output Parsers & Structured Output](../modules/03-output-parsers-structured-output.md).

## R

**RAG (Retrieval-Augmented Generation)**
A pattern that retrieves relevant documents from a knowledge source and injects them into the prompt so the model answers from up-to-date, grounded context rather than parametric memory alone. See [Module 6 — Retrieval & RAG](../modules/06-retrieval-and-rag.md).

**Reducer**
A function attached to a LangGraph state key (via `Annotated[..., reducer]`, e.g. `add_messages`) that defines how a node's returned value is merged into existing state instead of overwriting it. See [Module 9 — LangGraph Deep Dive](../modules/09-langgraph-deep-dive.md).

**Retriever**
A `Runnable` (`langchain_core.retrievers.BaseRetriever`) that takes a query string and returns relevant `Document`s; commonly produced by `vectorstore.as_retriever()`. See [Module 6 — Retrieval & RAG](../modules/06-retrieval-and-rag.md).

**Runnable**
The universal interface (`langchain_core.runnables.Runnable`) implemented by every LangChain component, providing `invoke`/`stream`/`batch` (plus async variants) and composition via `|`; the basis of LCEL. See [Module 4 — LCEL & the Runnable Interface](../modules/04-lcel-and-runnables.md).

**RunnableConfig**
The per-invocation config dict (`langchain_core.runnables.RunnableConfig`) carrying `callbacks`, `tags`, `metadata`, `configurable` values, and concurrency limits; threaded through every step of a chain or graph. See [Module 4 — LCEL & the Runnable Interface](../modules/04-lcel-and-runnables.md).

**RunnableLambda**
A wrapper (`langchain_core.runnables.RunnableLambda`) that turns an ordinary Python function into a `Runnable` so it can be composed in an LCEL chain. See [Module 4 — LCEL & the Runnable Interface](../modules/04-lcel-and-runnables.md).

**RunnableParallel**
A `Runnable` that runs several runnables on the same input concurrently and returns a dict of their results; also expressible as a plain dict literal inside an LCEL chain. See [Module 4 — LCEL & the Runnable Interface](../modules/04-lcel-and-runnables.md).

**RunnablePassthrough**
A `Runnable` that forwards its input unchanged (optionally adding keys via `.assign()`), commonly used to thread the original input alongside computed values in a chain. See [Module 4 — LCEL & the Runnable Interface](../modules/04-lcel-and-runnables.md).

## S

**Send API**
A LangGraph mechanism (`langgraph.types.Send`) for dynamic fan-out: a node returns one or more `Send(node, state)` objects to spawn parallel invocations of a node — the basis of map-reduce patterns. See [Module 9 — LangGraph Deep Dive](../modules/09-langgraph-deep-dive.md).

**State (LangGraph)**
The shared, typed data structure (a `TypedDict`, dataclass, or Pydantic model) that flows through a graph; nodes read it and return partial updates merged via reducers. See [Module 9 — LangGraph Deep Dive](../modules/09-langgraph-deep-dive.md).

**StateGraph**
The primary LangGraph builder class (`langgraph.graph.StateGraph`) to which you add nodes and edges, then `.compile()` into a runnable graph. See [Module 9 — LangGraph Deep Dive](../modules/09-langgraph-deep-dive.md).

**Streaming**
Incrementally emitting output as it is produced — token-by-token from a model (`stream`) or step-by-step from a graph — rather than waiting for the full result. See [Module 1 — Models](../modules/01-models-chat-and-llms.md) and [Module 4 — LCEL & the Runnable Interface](../modules/04-lcel-and-runnables.md).

**Structured output**
Model output constrained to a declared schema (Pydantic model, `TypedDict`, or JSON schema) so you get a typed object instead of free text, usually obtained via `with_structured_output`. See [Module 3 — Output Parsers & Structured Output](../modules/03-output-parsers-structured-output.md).

**SystemMessage**
The message type (`langchain_core.messages.SystemMessage`) that sets the model's role, instructions, and constraints, typically placed first in the message list. See [Module 1 — Models](../modules/01-models-chat-and-llms.md).

## T

**Temperature**
A sampling parameter (0–~1) controlling randomness of generation; lower is more deterministic/focused, higher is more varied. See [Module 1 — Models](../modules/01-models-chat-and-llms.md).

**Token**
The sub-word unit models read and generate; token counts drive context limits and billing, and are reported in `AIMessage.usage_metadata`. See [Module 1 — Models](../modules/01-models-chat-and-llms.md).

**Tool**
A callable exposed to a model with a name, description, and typed argument schema (created with the `@tool` decorator or `StructuredTool`) that the model may request via tool calling. See [Module 5 — Tools & Tool Calling](../modules/05-tools-and-tool-calling.md).

**Tool calling**
The mechanism by which a model emits structured `tool_calls` (name + arguments) on an `AIMessage`, which your code or a `ToolNode` then executes; enabled via `model.bind_tools([...])`. See [Module 5 — Tools & Tool Calling](../modules/05-tools-and-tool-calling.md).

**ToolMessage**
The message type (`langchain_core.messages.ToolMessage`) carrying a tool's execution result back to the model, linked to the originating call by `tool_call_id`. See [Module 5 — Tools & Tool Calling](../modules/05-tools-and-tool-calling.md).

**ToolNode**
A prebuilt LangGraph node (`langgraph.prebuilt.ToolNode`) that reads the latest `AIMessage`'s tool calls, runs the corresponding tools, and appends `ToolMessage`s to state. See [Module 8 — Agents with LangGraph](../modules/08-agents-with-langgraph.md).

**Trace**
A recorded tree of a run's steps (model calls, tool calls, chain spans) captured by LangSmith for debugging, latency analysis, and evaluation. See [Module 10 — Observability & Evaluation](../modules/10-observability-and-eval-langsmith.md).

## V

**Vector store**
A database (interface `langchain_core.vectorstores.VectorStore`; e.g. Chroma, FAISS, pgvector) that indexes embeddings and serves similarity search; the storage layer of RAG. See [Module 6 — Retrieval & RAG](../modules/06-retrieval-and-rag.md).

## W

**with_structured_output**
A method on chat models (`model.with_structured_output(Schema)`) that returns a runnable yielding instances of the given Pydantic/`TypedDict`/JSON schema instead of raw text — the preferred way to get structured results. See [Module 3 — Output Parsers & Structured Output](../modules/03-output-parsers-structured-output.md).

---

## Recap

- This glossary collects the load-bearing vocabulary of modern LangChain (v0.3+) and LangGraph, from low-level message types to high-level architectural patterns.
- **Core abstractions** (`Runnable`, `Chain`, `Document`, message types, `Embeddings`) live in `langchain-core`; **provider classes** live in partner packages; **agents/graphs** live in `langgraph`.
- When a term is ambiguous in everyday usage (e.g. *chunk*, *chain*), prefer the framework-specific meaning given here.
- Each entry links to the module where the concept is developed with runnable examples — use this appendix as an index, and the modules for depth.

See also: [Appendix A — Cheat Sheets](A-cheatsheets.md), [Appendix B — Common Errors & Fixes](B-common-errors.md), and [Appendix C — Versioning & Migration](C-versioning-and-migration.md).
