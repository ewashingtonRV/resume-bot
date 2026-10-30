# Task: Build AutoMango knowledge base in the vault

## Goal
Create wiki pages documenting my AutoMango work (career narrative + service documentation) so Remy can answer questions about it. All output goes in `vault/` and must follow the Vault Protocol in CLAUDE.md: YAML frontmatter on every file, `[[wiki links]]` cross-references, update `vault/index.md`, and append to `vault/log.md`.

## Structure rules (these clarify the Vault Protocol — do not create subdirectories)
1. **Narrative** — one page: `vault/projects/automango.md`. Two-tier layout: frontmatter + executive summary at top (Tier 1), then phase sections below (Tier 2).
2. **Services** — flat files in `vault/projects/` using suffix naming, three files per service:
   - `vault/projects/{service}-system-design.md`
   - `vault/projects/{service}-technical-description.md`
   - `vault/projects/{service}-business-context.md`
   Services: `rfi-assistant`, `rfp-automation`, `slide-selector`, `pitch`, `op1-ingest`, `client-template`, `aq-automation`, `dpm-feedback-model`.
3. Also create/update supporting nodes in `vault/tools/`, `vault/skills/`, and `vault/outcomes/` for any technology, discipline, or quantified impact you encounter in the sources, and cross-link them from the service pages.

## Voice and audience
- Write in **third person** ("Eric identified...", "Eric's team built..."). This is a wiki about Eric, retrieved by a chatbot.
- Audience is a recruiter or hiring manager talking to Remy. Lead with impact and clarity; keep deep technical detail in lower Tier-2 sections.
- Do not fabricate metrics, dates, or technical details not present in the sources.

## Handling gaps
My sources are incomplete. When a source doesn't cover something a file needs:
- Write what the sources support.
- Mark each gap inline with a consistent marker: `> TODO(gap): <what's missing and what source would fill it>`.
- If a source is unreachable (auth failure, dead link, unreadable file), do NOT guess its contents — create the file skeleton and mark the whole section as a TODO(gap).
- At the end, produce `vault/qa/automango-gap-report.md` consolidating every TODO(gap) as a checklist so I can supply missing info.

## The narrative (for vault/projects/automango.md)

### Phase 1 — Opportunity identification
- Eric interviewed 19 employees (16 hours of interview time) to understand the current sales/media operations process.
- He diagrammed the as-is process, identified automation opportunities, ranked them using the RICE framework, and presented findings to leadership.

### Phase 2 — Solo POCs
Working directly with stakeholders, Eric built the first three services himself: [[projects/rfi-assistant]], [[projects/rfp-automation]], [[projects/slide-selector]] (link to the suffix pages).

### Phase 3 — Team growth
Based on the POCs' success, leadership funded the initiative, named **AutoMango**. The team began as Eric plus a backend engineer and a frontend engineer. It has grown to an interdisciplinary team Eric leads: 1 project manager, 3 data scientists (1 full-time, 2 contractors, not counting Eric), 1 data engineer, and 4 engineers.

### Phase 4 — End-to-end sales automation
The team now automates the sales process end-to-end. Volume context: 500+ RFPs arrive in a 3–4 month window. The high-level workflow is in `/Users/ewashington/Desktop/media_ops_automation_workflow.md` (Mermaid) — use it to describe how the services connect:
- rfi-assistant and rfp-automation handle the initial steps of the sales process.
- Pitch (formerly Trendbot) is an MCP enabling faster, better product recommendations at the Kickoff meeting.
- op1-ingest and client-template automate post-kickoff Media Ops work.

### Phase 5 — Scope expansion beyond sales
Given the sales-automation success, scope expanded to other parts of the company: aq-automation and the dpm-feedback-model (just started).

## Service sources
For each service, use the listed source(s) to write the named file. Read sources; never modify them.

### rfi-assistant
- system_design: `/Users/ewashington/Desktop/github/tf-aws-automango/rvoh-automango-dev/us-east-1/rfi_pipeline.tf`
- technical_description: https://rvohealth.atlassian.net/wiki/spaces/MANGO/pages/3018358801/RFI+Assistant#Solution
- business_context: https://rvohealth.atlassian.net/wiki/spaces/MANGO/pages/3018358801/RFI+Assistant#Problem-Statement

### rfp-automation
- system_design: `/Users/ewashington/Desktop/rfp_pipeline_architecture.pdf`
- technical_description: https://rvohealth.atlassian.net/wiki/spaces/MANGO/pages/3002401628/RFP+Automation#Solution
- business_context: https://rvohealth.atlassian.net/wiki/spaces/MANGO/pages/3002401628/RFP+Automation#Problem-Statement

### slide-selector
- system_design: `/Users/ewashington/Desktop/github/tf-aws-automango/rvoh-automango-dev/us-east-1/slide_selector_pipeline.tf`
- technical_description: https://rvohealth.atlassian.net/wiki/spaces/MANGO/pages/3084222474/Slide+Selector#Solution
- business_context: https://rvohealth.atlassian.net/wiki/spaces/MANGO/pages/3084222474/Slide+Selector#Problem-Statement

### pitch (formerly Trendbot — note the alias in frontmatter tags so Remy can answer to either name)
- system_design:
  - Diagram: `/Users/ewashington/Desktop/tendbot_infra.pdf`
  - Markdown version of same diagram: `/Users/ewashington/Desktop/trendbot_mcp_architecture.md`
  - Layperson explanation: `/Users/ewashington/Desktop/trendbot_infra_explained.md`
- technical_description:
  - Data model powering the MCP: https://github.com/rvohealth/trendbot-v2-model/blob/main/snowflake/trendbot_v2_data_model_documentation.md
  - Overview + example queries: `/Users/ewashington/Desktop/Pitch_v2_Overview.pptx`
- business_context: `/Users/ewashington/Desktop/pitch_business_context.md`

### op1-ingest
- business_context (from my description; no external source):
  - Before automation: after the kickoff meeting, the Media Ops team took the product recommendations and targeting criteria, connected additional data points (e.g., how many "avails"/impressions could be offered to a partner, ad size, price), and manually entered everything into Operative 1 (the UI/database storing media plan information).
- technical_description (from my description + linked README):
  - The automation provides a UI that simplifies data entry by codifying business logic that was previously specialized knowledge held only by Media Ops.
  - An API takes UI input and retrieves avails: https://github.com/rvohealth/avails-api/blob/main/README.md
- system_design: no source provided — create skeleton with TODO(gap).

### client-template
- business_context (from my description + sources):
  - Before automation: in addition to Operative 1 entry, Media Ops re-entered media plan data into a client-provided spreadsheet. Tedious because every client sends a different template (different fields, column names, structure).
  - Example client template: https://docs.google.com/spreadsheets/d/15Ex6caakzuh3c4SnB_wF5QdVKJjB6NF2/edit?usp=sharing&ouid=116450298235039718327&rtpof=true&sd=true
  - Additional context: https://rvohealth.atlassian.net/wiki/spaces/MANGO/pages/3046637659/Client+Template+Automation#Problem-Statement
- technical_description (from my description + sources):
  - High-level solution: https://rvohealth.atlassian.net/wiki/spaces/MANGO/pages/3046637659/Client+Template+Automation#Solution
  - Flow detail: an inbound email from a team member contains an ID in the subject line plus the client's spreadsheet attachment. The ID is used to retrieve and join relevant data from Operative 1 warehouse tables in Databricks catalogs, then our fields are mapped to the client's spreadsheet fields.
  - Mapping code: https://github.com/rvohealth/automango/blob/jpacubas/dev/services/client-template-automation/src/field_mapper/main.py and https://github.com/rvohealth/automango/blob/jpacubas/dev/services/client-template-automation/src/field_mapper/data_dictionary.yaml
- system_design: very similar to rfp-automation; use `/Users/ewashington/Desktop/client_template_pipeline_architecture.pdf` and cross-link [[projects/rfp-automation-system-design]].

### aq-automation
- business_context: https://rvohealth.atlassian.net/wiki/spaces/MANGO/pages/3199533125/AQ+Automation#Problem-Statement
- technical_description: https://rvohealth.atlassian.net/wiki/spaces/MANGO/pages/3199533125/AQ+Automation#Solution
- system_design: https://rvohealth.atlassian.net/wiki/spaces/MANGO/pages/3199533125/AQ+Automation#Architecture

### dpm-feedback-model
- Just started. Single context source: https://docs.google.com/document/d/11voH5rZUqNSuj_deZWkAqWkQ6OtiJb0PupJppXHmD1E/edit?usp=sharing
- Create all three files, but mark status `in-progress` in frontmatter and expect heavy TODO(gap) usage.

## Order of operations
1. Read `vault/index.md` first (per protocol).
2. Gather all sources; note unreachable ones.
3. Write the narrative page, then service pages, then supporting tools/skills/outcomes nodes.
4. Cross-link everything; verify no dangling `[[wiki links]]`.
5. Write `vault/qa/automango-gap-report.md`.
6. Update `vault/index.md` and append to `vault/log.md`.
7. Finish with a summary: files created, links added, and the top gaps needing my input.
