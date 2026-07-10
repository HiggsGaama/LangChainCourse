---
layout: default
title: Prompts & Prompt Templates
---

# Module 2 — Prompts & Prompt Templates

Prompts are the contract between your application and the model. In a real system the prompt is not a string you type once — it is a *parameterized artifact* assembled at runtime from a stable instruction layer, user-supplied content, retrieved context, conversation history, and few-shot examples. LangChain's prompt template abstractions exist to make that assembly explicit, reusable, type-checked on its inputs, and composable with the rest of an [LCEL](04-lcel-and-runnables.md) chain.

This module covers string templates, chat (message) templates, `MessagesPlaceholder`, partials, few-shot prompting, example selectors, prompt composition, multimodal content, and pulling shared prompts from LangChain Hub. By the end you should be able to build prompts that keep the system instruction frozen while varying only what needs to vary — which matters both for correctness and, as you'll see in [Module 1](01-models-chat-and-llms.md) and [Module 6](06-retrieval-and-rag.md), for prompt-cache hit rates and cost.

> **Note:** All prompt template classes live in `langchain_core.prompts`. They are provider-agnostic — the same template feeds [Anthropic Claude](01-models-chat-and-llms.md), OpenAI, or any other chat model unchanged.

---

## Why templates exist

You *can* build a prompt with f-strings and pass the result to a model. For a one-off script that's fine. It stops being fine the moment any of the following is true:

- The same instruction is reused across many call sites and you want to change it in one place.
- Part of the prompt is stable (the system instruction) and part varies per request (the user's question). You want the stable part byte-identical across calls so it caches.
- The varying part comes from untrusted input and must be slotted into a fixed structure, not concatenated into instructions.
- You assemble messages from history, retrieved documents, and few-shot examples — each a different shape.
- You want the prompt to be a unit you can test, version, swap, and compose into a chain with `|`.

A `PromptTemplate` (or `ChatPromptTemplate`) is a declarative description of *what varies and where*. It declares its `input_variables`, validates that you supplied them, and produces a `PromptValue` that any model can consume. It is also a `Runnable`, so it slots directly into LCEL: `prompt | model | parser`.

```python
from langchain_core.prompts import ChatPromptTemplate

# The instruction is fixed; only {topic} changes per request.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise technical writer. Answer in at most three sentences."),
    ("human", "Explain {topic} to a senior engineer."),
])

print(prompt.input_variables)  # ['topic']
```

Separation of concerns is the headline benefit: the engineer who owns the system prompt and the code path that supplies `topic` are decoupled, and neither can accidentally corrupt the other.

---

## `PromptTemplate` vs `ChatPromptTemplate`

LangChain has two prompt families, matching the two model interfaces from [Module 1](01-models-chat-and-llms.md):

| Class | Produces | Use with |
|---|---|---|
| `PromptTemplate` | a single string (`StringPromptValue`) | text-completion LLMs (`BaseLLM`) |
| `ChatPromptTemplate` | a list of messages (`ChatPromptValue`) | chat models (`BaseChatModel`) — i.e. essentially everything today |

Modern Claude and GPT models are chat models, so **`ChatPromptTemplate` is what you'll use 95% of the time.** `PromptTemplate` still matters: it's the building block for the *content* of a single message, and it's what few-shot string templates and some legacy chains use.

### `PromptTemplate` (string)

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Summarize the following text in {n_words} words:\n\n{text}"
)

print(template.input_variables)          # ['n_words', 'text']
print(template.format(n_words=20, text="LangChain is a framework for LLM apps."))
# -> "Summarize the following text in 20 words:\n\nLangChain is a framework for LLM apps."
```

`from_template` infers `input_variables` from the `{...}` fields. You can also construct it explicitly when you want validation or a non-default format:

```python
template = PromptTemplate(
    template="Translate to {language}: {text}",
    input_variables=["language", "text"],
)
```

### `ChatPromptTemplate` (messages)

`ChatPromptTemplate.from_messages` takes a list where each entry becomes one message. The most common form is a `(role, template)` tuple:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant fluent in {language}."),
    ("human", "{question}"),
])

print(prompt.input_variables)  # ['language', 'question']

messages = prompt.format_messages(language="French", question="How do I say hello?")
for m in messages:
    print(type(m).__name__, "->", repr(m.content))
# SystemMessage -> 'You are a helpful assistant fluent in French.'
# HumanMessage  -> 'How do I say hello?'
```

Valid role strings are `"system"`, `"human"` (alias `"user"`), `"ai"` (alias `"assistant"`), and `"placeholder"` (covered below). Each tuple's second element is itself a string template, so any `{var}` inside it contributes to `input_variables`.

`from_template` on `ChatPromptTemplate` is a shortcut that creates a single human-message template:

```python
ChatPromptTemplate.from_template("What is {x}?")
# equivalent to ChatPromptTemplate.from_messages([("human", "What is {x}?")])
```

### Templating syntax: f-string, mustache, jinja2

The default `template_format` is `"f-string"` — single-brace `{var}` placeholders. To put a *literal* brace in an f-string template, double it: `{{` and `}}` (important when your prompt contains JSON examples). LangChain also supports `"mustache"` (`{{var}}`) and `"jinja2"`.

```python
from langchain_core.prompts import PromptTemplate

# Mustache — double braces, and you don't have to declare input_variables
mustache = PromptTemplate.from_template(
    "Hello {{name}}, your plan is {{plan}}.",
    template_format="mustache",
)
print(mustache.format(name="Alice", plan="Enterprise"))

# Jinja2 — full templating logic (loops, conditionals)
jinja = PromptTemplate.from_template(
    "Items:\n{% for item in items %}- {{ item }}\n{% endfor %}",
    template_format="jinja2",
)
print(jinja.format(items=["a", "b", "c"]))
```

> **⚠️ Gotcha:** Jinja2 is a full template language and **executes logic**. Never build a jinja2 template string from untrusted input — it's a sandbox-escape / SSTI risk. Use jinja2 only for templates *you* author. Mustache is logic-light and a good middle ground when your content is full of literal `{` braces (JSON, code) and the f-string doubling gets noisy. For most prompts, stick with the default f-string. See [Module 11 — Production & Deployment](11-production-and-deployment.md) for the broader prompt-injection discussion.

---

## Invoking a prompt: `PromptValue`, `.to_messages()`, and composition

Calling `.invoke()` (the `Runnable` interface) on a prompt returns a **`PromptValue`** — an intermediate object that can render itself either as a string or as a message list, so the *same* prompt value works whether it's handed to a chat model or a string LLM.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])

pv = prompt.invoke({"question": "What is the capital of France?"})
print(type(pv).__name__)   # ChatPromptValue
print(pv.to_messages())    # [SystemMessage(...), HumanMessage(...)]
print(pv.to_string())      # "System: You are a helpful assistant.\nHuman: What is the capital of France?"
```

You rarely create a `PromptValue` by hand — it exists so that piping a prompt into a model "just works". That piping is the whole point:

```python
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Provider-agnostic init. Swap "anthropic:claude-sonnet-4-6" for any provider:model id.
model = init_chat_model("anthropic:claude-sonnet-4-6")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant."),
    ("human", "Give me {n} tips for {topic}."),
])

chain = prompt | model | StrOutputParser()

print(chain.invoke({"n": 3, "topic": "writing clear commit messages"}))
# -> "1. ... 2. ... 3. ..."
```

`prompt | model | parser` is the canonical [LCEL](04-lcel-and-runnables.md) chain. The prompt renders inputs to a `PromptValue`, the model consumes it and returns an `AIMessage`, and the [output parser](03-output-parsers-structured-output.md) extracts the part you want. Because every link is a `Runnable`, the chain also supports `.batch()`, `.stream()`, and `.ainvoke()` for free.

Swapping providers is a one-line change — the prompt and parser are untouched:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4.1")
chain = prompt | model | StrOutputParser()  # same prompt, different model
```

---

## `MessagesPlaceholder`: injecting history and scratchpads

A `(role, template)` tuple produces exactly one message. But conversation history is a *variable-length list* of messages, and an agent's scratchpad is a list of tool calls and results. You can't express "drop a list of messages here" with a string tuple. That's what `MessagesPlaceholder` is for.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),     # a list of messages is spliced in here
    ("human", "{question}"),
])

messages = prompt.invoke({
    "history": [
        HumanMessage("My name is Alice."),
        AIMessage("Nice to meet you, Alice!"),
    ],
    "question": "What's my name?",
}).to_messages()

for m in messages:
    print(type(m).__name__, "->", m.content)
# SystemMessage -> You are a helpful assistant.
# HumanMessage  -> My name is Alice.
# AIMessage     -> Nice to meet you, Alice!
# HumanMessage  -> What's my name?
```

The placeholder's name (`"history"`) becomes an input variable expecting a list of `BaseMessage` objects. This is exactly how conversation memory plugs in — see [Module 7 — Memory & State](07-memory-and-state.md) — and how the agent "scratchpad" of intermediate tool steps is fed back to the model in [Module 8 — Agents with LangGraph](08-agents-with-langgraph.md).

### Shorthand and optional placeholders

The `("placeholder", "{history}")` tuple is equivalent to `MessagesPlaceholder("history")`:

```python
ChatPromptTemplate.from_messages([
    ("system", "..."),
    ("placeholder", "{history}"),   # same as MessagesPlaceholder("history")
    ("human", "{question}"),
])
```

By default a placeholder is **required** — invoking without it raises a `KeyError`. Mark it optional when history may be absent (e.g. the first turn of a conversation):

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history", optional=True),   # defaults to [] if not supplied
    ("human", "{question}"),
])

# Works without "history":
prompt.invoke({"question": "Hello!"})
```

> **✅ Best practice:** Put the system message first and keep it stable, then a `MessagesPlaceholder` for history, then the current human turn last. This ordering keeps the cacheable prefix (system + tools) byte-identical across turns, and is the layout every memory and agent integration expects.

---

## Partials: pre-filling some variables

Sometimes a variable is known at construction time, not call time — a tenant name, a fixed format instruction, or a value computed once. `.partial(...)` returns a new template with those variables bound, removing them from `input_variables`.

```python
from langchain_core.prompts import ChatPromptTemplate

base = ChatPromptTemplate.from_messages([
    ("system", "You are a support agent for {company}. Respond in {tone}."),
    ("human", "{question}"),
])

# Bind company once; callers only supply tone + question.
acme = base.partial(company="Acme Corp")
print(acme.input_variables)  # ['question', 'tone']

acme.invoke({"tone": "friendly", "question": "How do I reset my password?"})
```

### Partial with a callable for dynamic values

A partial value can also be a **zero-argument callable**, evaluated each time the template is formatted. This is the idiomatic way to inject "now" — the current date/time — without baking a stale timestamp into the template.

```python
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate

def current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Today's date is {today}."),
    ("human", "{question}"),
]).partial(today=current_date)   # pass the function, not current_date()

# Each invoke re-evaluates current_date()
prompt.invoke({"question": "What year is it?"})
```

> **⚠️ Gotcha:** A dynamic value like the current timestamp in the *system* message changes the cacheable prefix on every request, defeating prompt caching (see [Module 1](01-models-chat-and-llms.md)). If you need "today's date" *and* prompt caching, move the date into a later message (e.g. the human turn or a `MessagesPlaceholder`) so the stable system prefix stays byte-identical. A callable partial that lands in the prefix is convenient but cache-hostile.

---

## Few-shot prompting

Few-shot prompting steers the model by showing it worked examples of the input→output mapping you want. LangChain provides templates that assemble a block of examples into the prompt, in both string and chat flavors.

### `FewShotChatMessagePromptTemplate` (chat — preferred)

For chat models, render each example as a `human`/`ai` message pair. This reads to the model as a short prior conversation demonstrating the pattern.

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

# 1. The examples themselves — a list of dicts.
examples = [
    {"input": "2 + 2", "output": "4"},
    {"input": "2 ^ 3", "output": "8"},
    {"input": "10 - 7", "output": "3"},
]

# 2. How to render ONE example into messages.
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# 3. The few-shot block: applies example_prompt to every example.
few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 4. Assemble the final prompt: instructions + examples + the real question.
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a calculator. Reply with only the numeric answer."),
    few_shot,                       # the example pairs are spliced in here
    ("human", "{input}"),
])

for m in final_prompt.invoke({"input": "5 * 6"}).to_messages():
    print(type(m).__name__, "->", m.content)
# SystemMessage -> You are a calculator. Reply with only the numeric answer.
# HumanMessage  -> 2 + 2
# AIMessage     -> 4
# HumanMessage  -> 2 ^ 3
# AIMessage     -> 8
# HumanMessage  -> 10 - 7
# AIMessage     -> 3
# HumanMessage  -> 5 * 6
```

Note that `few_shot` itself exposes no input variables here — its examples are fixed — so `final_prompt.input_variables` is just `['input']`.

### `FewShotPromptTemplate` (string)

The string equivalent, for text-completion LLMs or when you're building the content of a single message. You supply an `example_prompt` (a `PromptTemplate`), a `prefix`, and a `suffix`:

```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

example_prompt = PromptTemplate.from_template("Word: {word}\nAntonym: {antonym}")

few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of every word.",
    suffix="Word: {word}\nAntonym:",   # the real input goes here
    input_variables=["word"],
)

print(few_shot.format(word="fast"))
# Give the antonym of every word.
#
# Word: happy
# Antonym: sad
#
# Word: tall
# Antonym: short
#
# Word: fast
# Antonym:
```

---

## Example selectors: dynamic few-shot

Hard-coding the same examples for every input wastes tokens and isn't always relevant. **Example selectors** choose *which* examples to include per input — for instance the semantically closest examples, or as many as fit a token budget. Pass an `example_selector` instead of `examples`.

### `SemanticSimilarityExampleSelector`

Selects the examples most similar to the current input, using embeddings and a vector store. This needs an embeddings model and a vector store (see [Module 6 — Retrieval & RAG](06-retrieval-and-rag.md) for both).

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

examples = [
    {"input": "What's the weather in Paris?", "output": "I'll check the forecast for Paris."},
    {"input": "How tall is Mount Everest?", "output": "Everest is 8,849 meters."},
    {"input": "Convert 10 km to miles.", "output": "10 km is about 6.21 miles."},
    {"input": "What's the capital of Japan?", "output": "Tokyo is the capital of Japan."},
]

# Build the selector. It embeds each example's text and stores it.
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(model="text-embedding-3-small"),
    InMemoryVectorStore,
    k=2,                               # return the 2 closest examples
    input_keys=["input"],              # embed using the "input" field
)

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot = FewShotChatMessagePromptTemplate(
    # No static `examples` — the selector picks them per input.
    example_selector=example_selector,
    example_prompt=example_prompt,
    input_variables=["input"],
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Mirror the style of the examples."),
    few_shot,
    ("human", "{input}"),
])

model = ChatAnthropic(model="claude-sonnet-4-6")
chain = final_prompt | model

# For a measurement question, the selector surfaces the Everest / km examples.
print(chain.invoke({"input": "How far is 5 miles in kilometers?"}).content)
```

> **Note:** `SemanticSimilarityExampleSelector` re-runs the similarity search on every format/invoke, so the few-shot block adapts to each input. The cost is one embedding call (for the query) plus a vector-store lookup per invocation.

### `LengthBasedExampleSelector`

Picks as many examples as fit within a maximum length, so long inputs leave room and short inputs get more shots. No embeddings required.

```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import LengthBasedExampleSelector

examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "fast", "antonym": "slow"},
    {"word": "hot", "antonym": "cold"},
]

example_prompt = PromptTemplate.from_template("Word: {word}\nAntonym: {antonym}")

selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25,            # approx word count of the formatted examples
)

dynamic = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every word.",
    suffix="Word: {word}\nAntonym:",
    input_variables=["word"],
)

# A short input leaves room for more examples; a very long input gets fewer.
print(dynamic.format(word="big"))
```

> **✅ Best practice:** Use `SemanticSimilarityExampleSelector` when relevance matters and you have many examples; use `LengthBasedExampleSelector` when you mainly care about staying within a token budget. Either way, dynamic few-shot keeps prompts lean compared to dumping every example into every request.

---

## Prompt composition and reuse

Prompts are composable objects, not just strings.

**Concatenate chat prompts with `+`.** Two `ChatPromptTemplate`s (or a template plus a message) combine into one:

```python
from langchain_core.prompts import ChatPromptTemplate

system = ChatPromptTemplate.from_messages([
    ("system", "You are a meticulous code reviewer."),
])
task = ChatPromptTemplate.from_messages([
    ("human", "Review this {language} code:\n\n{code}"),
])

review_prompt = system + task
print(review_prompt.input_variables)  # ['code', 'language']
```

This lets you keep a library of reusable system-prompt fragments and snap them together per use case — the same modular benefit as importing a function.

**`PipelinePromptTemplate` (legacy concept).** Older LangChain had `PipelinePromptTemplate` for composing a final template out of named sub-templates whose outputs feed each other. It is deprecated in modern LangChain; prefer plain composition (`+`, partials) or — for genuine multi-step assembly — an [LCEL](04-lcel-and-runnables.md) chain where each step formats part of the prompt. If you encounter `PipelinePromptTemplate` in older code, the migration is usually to build the pieces with `.partial(...)` and concatenation, or to compute the sub-prompts in a `RunnableLambda` and feed them into the final `ChatPromptTemplate`.

```python
# Modern equivalent of "compose sub-prompts": format pieces, then partial them in.
from langchain_core.prompts import ChatPromptTemplate

intro = "You are {persona}."
rules = "Always: {rule_1}. Never: {rule_2}."

system_text = f"{intro}\n{rules}"
prompt = ChatPromptTemplate.from_messages([
    ("system", system_text),
    ("human", "{question}"),
]).partial(persona="a tax advisor", rule_1="cite the relevant code section", rule_2="give legal guarantees")
```

---

## Multimodal content in templates

Chat prompt templates can carry images and other non-text content blocks alongside text. The content of a human message becomes a list of typed blocks; templated `{var}` fields can appear in those blocks too.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a vision assistant. Describe what you see."),
    ("human", [
        {"type": "text", "text": "{question}"},
        {"type": "image", "url": "{image_url}"},
    ]),
])

model = ChatAnthropic(model="claude-sonnet-4-6")
chain = prompt | model

chain.invoke({
    "question": "What is in this picture?",
    "image_url": "https://example.com/cat.png",
})
```

> **⚠️ Verify:** Multimodal content-block shapes have evolved across LangChain versions. Modern `langchain-core` uses a standard cross-provider format with flat keys (`{"type": "image", "url": ...}`, or `{"type": "image", "base64": ..., "mime_type": ...}` for inline bytes). An older transitional form used `"source_type": "url"`/`"base64"` (with `"data"` for the bytes), and some providers still accept the OpenAI-style `{"type": "image_url", "image_url": {"url": ...}}` shape. Confirm the exact block schema for your installed versions against the [langchain-core docs](https://python.langchain.com/) before relying on it. Image *inputs* are covered more fully in [Module 1 — Models](01-models-chat-and-llms.md).

---

## LangChain Hub: pulling shared prompts

LangChain Hub is a registry of community and team prompts you can pull by reference, so you don't re-author well-known prompts (e.g. ReAct agent prompts, RAG prompts). Pulling returns a ready-to-use prompt template.

```python
# Modern hub access ships with the `langsmith` SDK (a `langchain` dependency).
# The old standalone `langchainhub` package is deprecated.
from langchain import hub

# Pull a community prompt by "owner/name" reference.
prompt = hub.pull("rlm/rag-prompt")

print(prompt.input_variables)   # e.g. ['context', 'question']
# prompt is a ChatPromptTemplate you can drop straight into a chain:
# chain = prompt | model | parser
```

> **Note:** `hub.pull` makes a network call and (for non-public prompts) authenticates via your LangSmith credentials — see [Module 10 — Observability & Evaluation (LangSmith)](10-observability-and-eval-langsmith.md). For production, pin to a specific commit hash (`"owner/name:<hash>"`) so a prompt update upstream doesn't silently change your behavior, and consider vendoring critical prompts into your own repo rather than pulling at runtime.

---

## Best practices

- **Freeze the system prompt; parameterize only what varies.** A stable, byte-identical system message is both easier to reason about and a prerequisite for prompt caching. Push per-request values (dates, IDs, user content) into later messages, not the system prefix.
- **Validate inputs.** A template declares `input_variables`; missing keys raise immediately rather than producing a half-rendered prompt. Lean on this — don't `.partial()` away required validation by accident.
- **Keep untrusted text as data, not instructions.** When you slot user-supplied or retrieved text into a `{var}`, it lands as message *content*, not as part of your instruction layer. Still treat it as adversarial: a user can write "ignore previous instructions" inside `{question}`. Structural defenses (clear role separation, delimiting untrusted spans, not granting the model dangerous tools on untrusted input) matter — see the security discussion in [Module 11 — Production & Deployment](11-production-and-deployment.md). **Never** build a jinja2 template string from untrusted input.
- **Compose, don't copy-paste.** Reuse system fragments via `+` and `.partial(...)` instead of duplicating prompt text across call sites.
- **Use dynamic few-shot when examples are many or input-dependent.** Static few-shot is fine for a handful of fixed examples; reach for an example selector when relevance or token budget matters.
- **Cross-link the prompt to its consumers.** A prompt's `MessagesPlaceholder` ties it to [memory](07-memory-and-state.md) and [agents](08-agents-with-langgraph.md); its output feeds [output parsers](03-output-parsers-structured-output.md). Design the prompt with those downstream contracts in mind.

---

## Recap

- Templates exist for **reuse, separation of concerns, and a stable instruction layer** with parameterized variable content. They are `Runnable`s, so they compose with `|`.
- **`PromptTemplate`** renders a string; **`ChatPromptTemplate`** renders a message list. Use `from_template` / `from_messages`; inspect `.input_variables`. The default format is f-string (double `{{`/`}}` for literal braces); mustache and jinja2 are available, with jinja2 reserved for trusted templates only.
- `ChatPromptTemplate.from_messages` takes `(role, template)` tuples; **`MessagesPlaceholder`** (or `("placeholder", "{name}")`) splices in a variable-length list of messages — history, agent scratchpad — and can be marked `optional=True`.
- Invoking a prompt yields a **`PromptValue`** (`.to_messages()` / `.to_string()`), which lets `prompt | model | parser` work for any model.
- **`.partial(...)`** pre-binds variables; passing a **callable** (e.g. for the current date) evaluates it at format time — but a dynamic value in the system prefix breaks prompt caching.
- **Few-shot**: `FewShotChatMessagePromptTemplate` (chat) and `FewShotPromptTemplate` (string) assemble worked examples via an `example_prompt`.
- **Example selectors** make few-shot dynamic: `SemanticSimilarityExampleSelector` (embeddings + vector store, relevance-based) and `LengthBasedExampleSelector` (token-budget-based).
- Prompts **compose** with `+` and `.partial(...)`; `PipelinePromptTemplate` is legacy — prefer composition or LCEL. Templates support **multimodal** content blocks, and **LangChain Hub** lets you pull shared prompts.

---

## Exercises

1. **Build a parameterized assistant.** Create a `ChatPromptTemplate` with a fixed system message and a `{question}` human turn. Pipe it into `init_chat_model("anthropic:claude-sonnet-4-6")` and a `StrOutputParser`, then invoke it for three different questions. Verify `.input_variables` reports exactly `['question']`.

2. **Add conversation memory by hand.** Extend the template above with a `MessagesPlaceholder("history")` between the system and human messages. Manually maintain a list of `HumanMessage`/`AIMessage` pairs, invoke the chain twice, and confirm the model uses the earlier turn (e.g. recalls a name you gave it). Make the placeholder optional and confirm the first turn works with no history supplied.

3. **Dynamic date partial — and the cache trap.** Add a `{today}` field to the system message and bind it with a `current_date` callable via `.partial(...)`. Then refactor so the date lands in the *human* turn instead of the system prefix. Explain in a comment why the second version is friendlier to prompt caching.

4. **Static few-shot classifier.** Using `FewShotChatMessagePromptTemplate`, build a sentiment classifier that, given 4 labeled examples (`positive`/`negative`/`neutral`), classifies a new sentence. Inspect the rendered messages with `.invoke({...}).to_messages()` and confirm the example pairs appear before the real input.

5. **Dynamic few-shot with semantic selection.** Convert Exercise 4 to use `SemanticSimilarityExampleSelector` with `OpenAIEmbeddings` and `InMemoryVectorStore`, `k=2`. Provide ~8 examples and confirm that different inputs surface different example subsets (print the selected examples).

6. **Compose and pull.** Build two reusable `ChatPromptTemplate` fragments (a persona system fragment and a task fragment) and combine them with `+`. Separately, `hub.pull("rlm/rag-prompt")` and print its `input_variables`. Compare the two approaches and note when you'd vendor a prompt into your repo versus pulling it at runtime.
