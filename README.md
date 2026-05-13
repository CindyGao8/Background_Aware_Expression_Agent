````markdown
# Background-Aware Expression Agent

A role-aware, document-grounded RAG system that explains the same project differently based on the user's background, stakeholder role, skill level, and communication needs.

The goal of this project is not only to answer a question correctly, but to express the answer in the most useful way for different audiences, such as engineers, product managers, business stakeholders, and general users.

## Overview

Standard RAG systems can retrieve project documents and generate answers, but they often produce one generic explanation for every user. In real technical teams, different stakeholders need different explanations of the same project.

The **Background-Aware Expression Agent** separates:

- **What to say**: controlled by retrieved project documents.
- **How to say it**: controlled by user background, selected role, and the expression layer.

The system follows the principle:

> RAG decides what the system is allowed to say; the expression layer decides how that evidence should be communicated.

## Key Features

- **Streamlit user interface** for uploading files, selecting roles, asking questions, and inspecting outputs.
- **Unique user ID** for separating each user's profile, background memory, uploaded documents, and retrieval index.
- **Project evidence vs. background personalization boundary**:
  - Project documents are used as the factual source of truth.
  - Resume/background files are used only for personalization, such as tone, technical depth, and jargon level.
- **Hybrid retrieval pipeline**:
  - BM25 sparse retrieval for exact keyword matching.
  - Vector search for semantic retrieval.
  - FAISS for local vector indexing.
  - Zilliz Cloud for persistent vector storage in the cloud deployment path.
  - Cross-encoder reranking for better candidate ordering.
- **Grounded base-answer generation** with citations.
- **Expression layer** for role-aware rewriting.
- **GPT-4o-based fine-tuned rewriting behavior** using role-aware instruction examples.
- **Prompt engineering and ExpressionPlan control** for role, tone, jargon level, structure, and grounding policy.
- **Debugging and transparency tools**, including retrieved chunks, citations, routing metadata, base answer, expression plan, and final rewritten answer.
- **Evaluation framework** for concept coverage, role fit, source isolation, citation support, and hallucination safety.

## System Architecture

The system is organized as a multi-layer pipeline:

```text
User query + role + uploaded files
        ↓
Streamlit interface with unique user ID
        ↓
User-specific profile, memory, document store, and retrieval index
        ↓
Query orchestrator
        ↓
Hybrid retrieval: BM25 + vector search + reranker
        ↓
Grounded base answer with citations
        ↓
ExpressionPlan construction
        ↓
Role-aware rewriting
        ↓
Final answer + citations + debug trace
````

## Project Structure

```bash
background-aware-expression-agent/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── core/
│   ├── query_orchestrator.py
│   └── expression_layer.py
│
├── memory/
│   └── background_memory.py
│
├── retrieval/
│   └── rag_system.py
│
├── data/
│   └── sample_docs/
│
├── eval_outputs/
│   ├── evaluation_report.md
│   ├── evaluation_results.json
│   └── evaluation_summary.csv
│
├── tests/
│   └── test_imports.py
│
└── evaluate_project_agent.py
```

## Main Components

### 1. Streamlit Interface

The Streamlit app provides the user-facing workflow. Users can:

* enter or create a user ID,
* upload project documents,
* upload a resume or background file,
* select a stakeholder role,
* ask natural-language questions,
* view the generated answer,
* inspect citations and debug information.

### 2. User-Specific Memory

Each user is associated with a unique user ID. The ID is used to organize:

* user profile,
* structured background memory,
* vector background memory,
* uploaded project document store,
* retrieval index.

This prevents different users' project files or background profiles from being mixed together.

### 3. Data Boundary

The system separates two types of data:

| Data Type               | Purpose                             |
| ----------------------- | ----------------------------------- |
| Project documents       | Factual source of truth for answers |
| Resume/background files | Personalization only                |

Project documents determine what claims the system can make. Background memory only controls how the answer is expressed.

### 4. Query Orchestrator

The query orchestrator creates a structured understanding of the user request. It identifies:

* query type,
* target topic,
* selected or inferred role,
* whether project context is needed,
* whether background personalization is needed,
* whether clarification may be needed,
* routing confidence and risk flags.

### 5. Hybrid Retrieval

The retrieval layer combines sparse retrieval, dense retrieval, and reranking.

* **BM25** handles exact terms, section names, file names, and metrics.
* **Vector search** handles semantic similarity.
* **FAISS** is used for local vector indexing and testing.
* **Zilliz Cloud** is used for persistent vector storage in the cloud deployment path.
* **Cross-encoder reranking** improves the final ordering of retrieved chunks.

Each retrieved chunk keeps metadata such as source file, page or section, and chunk ID for citation display.

### 6. Expression Layer

The expression layer is the core contribution of the project. It converts a grounded base answer into a role-aware explanation.

The layer combines:

* supervised fine-tuning,
* prompt engineering,
* structured expression planning,
* role-specific rewriting policies,
* grounded rewriting constraints.

The fine-tuning data was organized as supervised instruction pairs:

```text
(role, base_answer, instruction, citations) -> role_aware_rewrite
```

The fine-tuning objective is not factual knowledge injection. Project facts are still supplied through retrieval at inference time. Fine-tuning teaches the model a more consistent role-aware rewriting pattern.

At inference time, the system builds an `ExpressionPlan` that controls:

* target role,
* explanation depth,
* tone,
* jargon policy,
* output structure,
* emphasis,
* de-emphasis,
* grounding policy.

### 7. Role-Specific Output

| Role            | Expression Focus                                                 |
| --------------- | ---------------------------------------------------------------- |
| General         | Simple explanation, core idea, limited jargon                    |
| Engineer        | Architecture, modules, data flow, APIs, tradeoffs, failure modes |
| Product Manager | Workflow, dependencies, handoffs, risks, success criteria        |
| Business User   | Value, feasibility, operational impact, risks, decision points   |

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/CindyGao8/Background_Aware_Expression_Agent.git
cd Background_Aware_Expression_Agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file or use Streamlit secrets.

Example `.env`:

```bash
OPENAI_API_KEY=your_openai_api_key
ZILLIZ_URI=your_zilliz_cloud_uri
ZILLIZ_TOKEN=your_zilliz_cloud_token
```

For Streamlit Community Cloud, configure secrets in the app settings:

```toml
OPENAI_API_KEY = "your_openai_api_key"
ZILLIZ_URI = "your_zilliz_cloud_uri"
ZILLIZ_TOKEN = "your_zilliz_cloud_token"
```

Do not hard-code API keys in the repository.

## How to Run

### Run the Streamlit app locally

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

### Run evaluation

```bash
python evaluate_project_agent.py
```

Evaluation outputs are saved under:

```bash
eval_outputs/
```

## Example Usage

### Example 1: Engineer Explanation

**User role:** Engineer

**Query:**

```text
Explain how this project works from an engineering perspective.
```

**Expected behavior:**

The system retrieves project evidence and rewrites the answer with emphasis on:

* system architecture,
* modules,
* data flow,
* retrieval pipeline,
* implementation tradeoffs,
* failure modes.

### Example 2: Product Manager Explanation

**User role:** Product Manager

**Query:**

```text
Explain this project for a PM who needs to understand workflow and risks.
```

**Expected behavior:**

The system emphasizes:

* user workflow,
* dependencies,
* handoffs,
* implementation risks,
* success criteria.

### Example 3: Business Explanation

**User role:** Business User

**Query:**

```text
What is the business value of this project?
```

**Expected behavior:**

The system explains:

* business value,
* feasibility,
* operational impact,
* risks,
* decision points,
* documented limitations.

If the uploaded documents do not provide evidence for ROI, deployment, or production readiness, the system should state that the information is not specified.

### Example 4: General Explanation

**User role:** General

**Query:**

```text
What is this project about?
```

**Expected behavior:**

The system gives a clear, simple explanation with limited technical jargon.

## Evaluation

The system compares two settings:

1. **Baseline RAG**: retrieves project evidence and generates an answer without role-aware rewriting.
2. **Role-aware agent**: uses the same retrieval pipeline but adds expression planning and role-aware rewriting.

The evaluation uses five dimensions:

| Metric               | Meaning                                                         |
| -------------------- | --------------------------------------------------------------- |
| Concept Coverage     | Whether the answer includes expected project concepts           |
| Role Fit             | Whether the answer matches the target stakeholder               |
| Source Isolation     | Whether citations come only from allowed uploaded project files |
| Citation Support     | Whether the answer preserves evidence links                     |
| Hallucination Safety | Whether unsupported claims are avoided                          |

The overall score is:

```text
Overall =
0.30 * ConceptCoverage
+ 0.25 * RoleFit
+ 0.20 * SourceIsolation
+ 0.10 * CitationSupport
+ 0.15 * HallucinationSafety
```

The evaluation combines rule-based checks with manual review. Human reviewers inspect whether citations support the generated claims, whether the role-specific framing is useful, and whether the answer avoids unsupported assumptions.

## Deployment

The application was deployed through **Streamlit Community Cloud** as a browser-accessible prototype.

The deployment includes:

* user ID input,
* project-document upload,
* resume/background upload,
* role selection,
* RAG indexing,
* role-aware answer generation,
* citation display,
* debug inspection.

For persistent vector storage in the cloud deployment path, the system uses **Zilliz Cloud**. API keys and environment variables are managed through **Streamlit secrets** rather than being hard-coded in the repository.

The system is still a prototype. A full production version would require stronger authentication, role-based access control, production-grade monitoring, and human-review escalation workflows.

## Limitations

* The evaluation set is small.
* The scoring framework is lightweight and partially rule-based.
* Citation support is measured at a coarse level.
* The current deployment is a prototype rather than a full enterprise production system.
* Future work should expand the test set, improve citation-faithfulness evaluation, support user-profile editing, and strengthen production controls.

## Core Design Principle

The main contribution of this project is the separation between factual grounding and communicative expression:

> Project RAG controls what the system can claim.
> The expression layer controls how the answer is communicated.

```
```
