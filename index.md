---
layout: default
title: The In-Depth LangChain Course for LLM Engineers
---

# The In-Depth LangChain Course for LLM Engineers

A rigorous, code-first course on **modern LangChain** (the `langchain-core` / partner-package era, v0.3+) and **LangGraph**. Written for engineers who already know Python and want to master LangChain as a tool for building production LLM applications.

> This course teaches the **current** way to build with LangChain: LCEL & the Runnable interface for composition, `.with_structured_output()` for typed results, and **LangGraph** for agents (the modern replacement for the legacy `AgentExecutor`). Legacy APIs are pointed out and labeled as such so you can read older code without adopting it.

---

## How to read this course

Every module is a standalone Markdown file in [`modules/`](modules/) and [`appendix/`](appendix/). They render beautifully in VS Code (open the preview with `Ctrl+Shift+V`), on GitHub, on GitHub Pages, or in Obsidian. Code blocks are copy-paste ready.

---

## Environment setup

Python 3.10+ is assumed. Install the core packages plus whichever providers you use:

```bash
# Core framework + the LCEL/Runnable primitives
pip install -U langchain langchain-core

# Provider integrations (pick what you need)
pip install -U langchain-anthropic     # Claude models (primary in this course)
pip install -U langchain-openai        # OpenAI models (shown as the swap-in alternative)

# Agents & stateful graphs
pip install -U langgraph

# Community integrations (loaders, vector stores, tools, etc.)
pip install -U langchain-community

# Observability & evaluation (optional but recommended)
pip install -U langsmith

# For the Advanced & Specialized modules (13-16), as needed:
pip install -U langchain-mcp-adapters          # Module 15 — MCP
pip install -U langgraph-supervisor langgraph-swarm  # Module 14 — multi-agent prebuilts
pip install -U presidio-analyzer presidio-anonymizer # Module 13 — optional PII redaction
```

Set your API keys as environment variables (the course defaults to Claude):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."          # only if you use OpenAI examples

# Optional: turn on LangSmith tracing for any run
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="lsv2_..."
```

> **Note:** Most examples use `init_chat_model("anthropic:claude-sonnet-4-6")`, which is provider-agnostic — swap the string to `"openai:gpt-4.1"` (or any supported provider) without changing the rest of the code. Current Claude model IDs used here: `claude-opus-4-8` (most capable), `claude-sonnet-4-6` (balanced default), `claude-haiku-4-5` (fast/cheap).

---

## Curriculum

### Foundations
| # | Module | What you'll learn |
|---|--------|-------------------|
| 0 | [Orientation & The Ecosystem](docs/modules/00-orientation-and-ecosystem) | What LangChain is/isn't, the package layout, the Runnable mental model, how to not get burned by versioning |
| 1 | [Models: Chat Models & LLMs](docs/modules/01-models-chat-and-llms) | Messages, `invoke`/`stream`/`batch`/async, params, multimodal, caching, rate limits, fallbacks |
| 2 | [Prompts & Prompt Templates](docs/modules/02-prompts) | `ChatPromptTemplate`, `MessagesPlaceholder`, few-shot, example selectors, composition |
| 3 | [Output Parsers & Structured Output](docs/modules/03-output-parsers-structured-output) | `StrOutputParser`, Pydantic, `.with_structured_output()`, tool-based extraction |
| 4 | [LCEL & the Runnable Interface](docs/modules/04-lcel-and-runnables) | The Runnable protocol, `|` composition, parallel/branch/lambda, config, streaming, custom runnables |

### Building Blocks
| # | Module | What you'll learn |
|---|--------|-------------------|
| 5 | [Tools & Tool Calling](docs/modules/05-tools-and-tool-calling) | `@tool`, arg schemas, binding tools, parsing tool calls, toolkits, error handling |
| 6 | [Retrieval & RAG](docs/modules/06-retrieval-and-rag) | Loaders, splitters, embeddings, vector stores, retrievers, RAG chains, advanced RAG |
| 7 | [Memory & Conversation State](docs/modules/07-memory-and-state) | Message history, `RunnableWithMessageHistory`, LangGraph persistence, trimming/summarizing |

### Agents & LangGraph
| # | Module | What you'll learn |
|---|--------|-------------------|
| 8 | [Agents with LangGraph](docs/modules/08-agents-with-langgraph) | Why LangGraph, `create_react_agent`, tool-calling loops, human-in-the-loop, streaming |
| 9 | [LangGraph Deep Dive](docs/modules/09-langgraph-deep-dive) | `StateGraph`, reducers, edges/cycles, subgraphs, checkpointers, time travel, `Send`/map-reduce |

### Production
| # | Module | What you'll learn |
|---|--------|-------------------|
| 10 | [Observability & Eval (LangSmith)](docs/modules/10-observability-and-eval-langsmith) | Tracing, datasets, evaluators, eval-driven development, prompt management |
| 11 | [Production & Deployment](docs/modules/11-production-and-deployment) | Serving, streaming to clients, async/concurrency, caching, cost/tokens, callbacks, security |

### Projects
| # | Module | What you'll build |
|---|--------|-------------------|
| 12 | [Capstone Projects](docs/modules/12-capstone-projects) | A cited RAG chatbot, a multi-tool human-in-the-loop agent, and a SQL agent |

### Advanced & Specialized
| # | Module | What you'll learn |
|---|--------|-------------------|
| 13 | [Security, Safety & Guardrails](docs/modules/13-security-and-guardrails) | OWASP LLM Top 10, prompt injection & the lethal trifecta, input/output guardrails, PII redaction, moderation, tool/agent sandboxing & HITL gates, multi-tenant RAG access control, guardrail frameworks, red-teaming |
| 14 | [Multi-Agent Systems](docs/modules/14-multi-agent-systems) | Supervisor / swarm / hierarchical patterns, `Command` handoffs, `langgraph-supervisor` & `langgraph-swarm`, shared state, when *not* to go multi-agent |
| 15 | [MCP & Interoperability](docs/modules/15-mcp-and-interoperability) | Model Context Protocol, `langchain-mcp-adapters`, wiring MCP servers into agents, exposing your own, and MCP security |
| 16 | [Prompt Engineering & Agentic Patterns](docs/modules/16-prompt-engineering-and-agentic-patterns) | CoT/decomposition/self-consistency, the agentic patterns (routing, parallelization, orchestrator-workers, evaluator-optimizer, reflection), workflow-vs-agent, and a Prompting-vs-RAG-vs-Fine-tuning decision guide |

### Appendix
- [A · Cheat Sheets](docs/appendix/A-cheatsheets) — imports, Runnable methods, common patterns at a glance
- [B · Common Errors & Fixes](docs/appendix/B-common-errors) — the exceptions you'll actually hit, and how to resolve them
- [C · Versioning & Migration](docs/appendix/C-versioning-and-migration) — v0.1 → v0.2 → v0.3, deprecations, staying current
- [D · Glossary](docs/appendix/D-glossary) — every term used in the course, defined

---

## How to use it well

1. **Run the code.** Every snippet is runnable. Keep a scratch file open and execute as you read.
2. **Do the exercises.** Each module ends with hands-on tasks — they're where the learning sticks.
3. **Follow the order for foundations (0–4), then dip into building blocks as needed.** Agents (8–9) assume you're comfortable with tools (5) and LCEL (4).

Happy building. 🦜🔗
