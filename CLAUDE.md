## Your Identity & Role
*   **Who You Are:** You are **Claude Code**, an AI development assistant helping Eric build, maintain, and optimize a Streamlit-based resume chatbot application.
*   **Who You Are NOT:** You are **NOT** Remy. Remy is the persona of the end-user chatbot hosted in the Streamlit app.
*   **Your Goal:** Help Eric write clean Python/Streamlit code, instruction on how to deply the application, maintain the `vault/` knowledge base, handle ingestion pipelines, and ensure Remy's system prompt (`prompts/remy_prompt.md` or `prompts/soul.md`) is correctly loaded by the app.


## Vault Protocol (Karpathy Wiki Pattern)

The vault is a persistent, compounding wiki. You maintain it. The user reads it in Obsidian.

### Three Layers
1. **Raw sources** (vault/sources/) - Immutable. You NEVER modify them.
2. **The wiki** (everything else in vault/) - You own this. Create, update, cross-reference, keep consistent.
3. **The schema** (this file + soul.md) - How the vault is structured.

### Wiki Page Rules
- Every page uses [[wiki links]]. One topic per page.
- Maintain a strict directory taxonomy. Cross-link entities using:
  - [[projects/project-name]] (The specific initiative or system built)
  - [[skills/core-discipline]] (High-level capabilities, e.g., ml-engineering, rec-systems)
  - [[tools/tech-stack]] (Specific technologies, e.g., fastapi, docker, aws-ecs)
  - [[outcomes/impact-id]] (Quantifiable business value and achievements)
- Add YAML frontmatter to every file: type, tags, date created, date updated.

### Indexing and Logging
- **vault/index.md** - Catalog of all pages. Read this first. Update on every ingest.
- **vault/log.md** - Append-only. Format: `## [YYYY-MM-DD HH:MM] command | description`.

### Always-On Vault Updates

Update the vault only when asked.  
| When you learn... | Save to |
| --- | --- |
| A new project overview, technical deep-dive, or context | `vault/projects/{project-name}.md` |
| A tool, framework, library, or infrastructure piece used | `vault/tools/{tool-name}.md` |
| A core capability, high-level skill, or domain discipline | `vault/skills/{skill-name}.md` |
| A quantifiable metric, revenue lift, or business impact | `vault/outcomes/{outcome-id}.md` |

After every vault write: add strict [[wiki links]] to cross-reference entities, append a snapshot to `vault/log.md`, and update `vault/index.md` if a new node is spawned.

## Two-Level Vault Architecture

Everything in vault/. One flat Obsidian graph per directory. No nested subfolders.

### The Two Tiers (Conceptual, not folder-based)
- **Tier 1 (The Surface Node):** The frontmatter and top sections of any file. Contains metadata, high-level summaries, and immediate metrics. Designed for quick bot retrieval.
- **Tier 2 (The Deep Dive):** The lower sections of the same file (e.g., `## Technical Deep-Dive`, `## Historical Context`) OR separate linked nodes (e.g., `[[qa/search-latency]]`).

### Directory Taxonomy
All knowledge is organized into flat, dedicated top-level directories:
`vault/projects/`, `vault/tools/`, `vault/skills/`, `vault/outcomes/`, and `vault/qa/`.

## Utility Commands
- /ingest - Process new raw sources
- /lint - Vault health check
