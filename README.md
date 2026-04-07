# Background-Aware Expression Agent

A modular multi-agent system that explains the same underlying content differently based on the user's background, role, skill level, and communication needs.

The goal of this project is not just to answer a question correctly, but to express the answer in the most suitable way for different audiences such as product managers, engineers, business stakeholders, or non-technical users.

---

## Overview

The **Background-Aware Expression Agent** is designed to:

- understand who the user is
- retrieve relevant background and project context
- interpret the user query
- generate a correct base explanation
- adapt the explanation into different styles depending on the user profile

For example, the system can explain the same concept:
- to a **PM** with more workflow, dependency, and risk emphasis
- to an **engineer** with more architecture, implementation, and tradeoff details
- to a **business stakeholder** with more focus on value, outcomes, and decisions
- to a **beginner** with simpler language, fewer technical terms, and more analogies

---

## Project Structure

```bash
background-aware-expression-agent/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── app/
│   ├── main.py
│   ├── config.py
│   └── constants.py
│
├── data/
│   ├── sample_users/
│   ├── sample_projects/
│   └── demo_inputs/
│
├── docs/
│   ├── architecture.md
│   ├── pipeline.md
│   ├── team_roles.md
│   └── demo_script.md
│
├── memory/
│   ├── background_parser.py
│   ├── chunker.py
│   ├── profile_store.py
│   ├── vector_store.py
│   └── memory_manager.py
│
├── retrieval/
│   ├── background_retriever.py
│   ├── project_retriever.py
│   ├── embedder.py
│   └── reranker.py
│
├── agents/
│   ├── query_understanding_agent.py
│   ├── clarification_agent.py
│   ├── response_router_agent.py
│   ├── base_explainer_agent.py
│   ├── expression_planner_agent.py
│   └── expression_rewriter_agent.py
│
├── prompts/
│   ├── background_parser.txt
│   ├── query_understanding.txt
│   ├── ambiguity_detection.txt
│   ├── base_explainer.txt
│   ├── expression_planner.txt
│   └── expression_rewriter.txt
│
├── pipelines/
│   ├── onboarding_pipeline.py
│   ├── query_pipeline.py
│   └── orchestrator.py
│
├── models/
│   ├── llm_client.py
│   ├── embedding_client.py
│   └── schemas.py
│
├── frontend/
│   ├── app.py
│   └── ui_helpers.py
│
├── evaluation/
│   ├── eval_retrieval.py
│   ├── eval_expression.py
│   ├── eval_personalization.py
│   └── sample_eval_set.json
│
└── tests/
    ├── test_background_parser.py
    ├── test_query_understanding.py
    ├── test_retrieval.py
    ├── test_expression.py
    └── test_pipeline.py
