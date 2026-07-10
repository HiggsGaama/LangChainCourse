// Course manifest — drives the sidebar and prev/next navigation.
// Each item.path is relative to index.html (the langchain-course/ folder).
window.COURSE_MANIFEST = [
  {
    group: "Foundations",
    items: [
      { path: "modules/00-orientation-and-ecosystem.md",        title: "0 · Orientation & The Ecosystem" },
      { path: "modules/01-models-chat-and-llms.md",             title: "1 · Models: Chat Models & LLMs" },
      { path: "modules/02-prompts.md",                          title: "2 · Prompts & Prompt Templates" },
      { path: "modules/03-output-parsers-structured-output.md", title: "3 · Output Parsers & Structured Output" },
      { path: "modules/04-lcel-and-runnables.md",               title: "4 · LCEL & the Runnable Interface" }
    ]
  },
  {
    group: "Building Blocks",
    items: [
      { path: "modules/05-tools-and-tool-calling.md",           title: "5 · Tools & Tool Calling" },
      { path: "modules/06-retrieval-and-rag.md",                title: "6 · Retrieval & RAG" },
      { path: "modules/07-memory-and-state.md",                 title: "7 · Memory & Conversation State" }
    ]
  },
  {
    group: "Agents & LangGraph",
    items: [
      { path: "modules/08-agents-with-langgraph.md",            title: "8 · Agents with LangGraph" },
      { path: "modules/09-langgraph-deep-dive.md",              title: "9 · LangGraph Deep Dive" }
    ]
  },
  {
    group: "Production",
    items: [
      { path: "modules/10-observability-and-eval-langsmith.md", title: "10 · Observability & Eval (LangSmith)" },
      { path: "modules/11-production-and-deployment.md",        title: "11 · Production & Deployment" }
    ]
  },
  {
    group: "Projects",
    items: [
      { path: "modules/12-capstone-projects.md",                title: "12 · Capstone Projects" }
    ]
  },
  {
    group: "Advanced & Specialized",
    items: [
      { path: "modules/13-security-and-guardrails.md",          title: "13 · Security, Safety & Guardrails" },
      { path: "modules/14-multi-agent-systems.md",              title: "14 · Multi-Agent Systems" },
      { path: "modules/15-mcp-and-interoperability.md",         title: "15 · MCP & Interoperability" },
      { path: "modules/16-prompt-engineering-and-agentic-patterns.md", title: "16 · Prompt & Agentic Patterns" }
    ]
  },
  {
    group: "Appendix",
    items: [
      { path: "appendix/A-cheatsheets.md",                      title: "A · Cheat Sheets" },
      { path: "appendix/B-common-errors.md",                    title: "B · Common Errors & Fixes" },
      { path: "appendix/C-versioning-and-migration.md",         title: "C · Versioning & Migration" },
      { path: "appendix/D-glossary.md",                         title: "D · Glossary" }
    ]
  }
];
