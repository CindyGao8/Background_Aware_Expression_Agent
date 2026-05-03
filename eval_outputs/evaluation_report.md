# Project Agent Evaluation Report

## Evaluation setup

- Docs directory: `data/sample_docs/uploaded_projects/demo_user`
- Allowed project files: Documentation - Casual Impact + Transformer - Han.pdf
- Number of test cases: 9
- Systems compared: baseline RAG vs role-aware agent

## Aggregate scores

- Average overall score across all runs: **0.892**
- Average source isolation across all runs: **1.000**
- Average role fit across all runs: **0.758**
- Average hallucination safety across all runs: **1.000**

## Baseline comparison

| System | Overall score | Role fit | Source isolation | Hallucination safety |
|---|---:|---:|---:|---:|
| Baseline RAG | 0.833 | 0.586 | 1.000 | 1.000 |
| Role-aware agent | 0.951 | 0.931 | 1.000 | 1.000 |

The baseline uses the same retrieval system but disables the expression layer. The role-aware agent uses the full expression layer to adapt the same retrieved evidence for the target audience while preserving document grounding.

## Test results

### general_summary — baseline_rag (target role: general)

- Overall score: **1.0**
- Concept coverage: 1.0
- Role fit: 1.0
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 4, 5, 6, 7]

### general_summary — role_aware_agent (target role: general)

- Overall score: **0.9571**
- Concept coverage: 0.8571
- Role fit: 1.0
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 4, 5, 6, 7]

### engineer_explanation — baseline_rag (target role: engineer)

- Overall score: **0.9083**
- Concept coverage: 0.7778
- Role fit: 0.9
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 3, 7, 8, 9]

### engineer_explanation — role_aware_agent (target role: engineer)

- Overall score: **1.0**
- Concept coverage: 1.0
- Role fit: 1.0
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 3, 7, 8, 9]

### engineer_workflow — baseline_rag (target role: engineer)

- Overall score: **0.8821**
- Concept coverage: 0.8571
- Role fit: 0.7
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 3, 4, 6, 7]

### engineer_workflow — role_aware_agent (target role: engineer)

- Overall score: **1.0**
- Concept coverage: 1.0
- Role fit: 1.0
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 3, 4, 6, 7]

### engineer_model_roles — baseline_rag (target role: engineer)

- Overall score: **0.875**
- Concept coverage: 1.0
- Role fit: 0.5
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 3, 4, 6, 7, 8, 9]

### engineer_model_roles — role_aware_agent (target role: engineer)

- Overall score: **1.0**
- Concept coverage: 1.0
- Role fit: 1.0
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 3, 4, 6, 7, 8, 9]

### engineer_evaluation_results — baseline_rag (target role: engineer)

- Overall score: **0.725**
- Concept coverage: 0.6667
- Role fit: 0.3
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 3, 4, 5, 6, 7]

### engineer_evaluation_results — role_aware_agent (target role: engineer)

- Overall score: **0.9667**
- Concept coverage: 0.8889
- Role fit: 1.0
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 3, 4, 5, 6, 7]

### pm_explanation — baseline_rag (target role: pm)

- Overall score: **0.8625**
- Concept coverage: 0.75
- Role fit: 0.75
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 3, 4, 5, 6, 7]

### pm_explanation — role_aware_agent (target role: pm)

- Overall score: **1.0**
- Concept coverage: 1.0
- Role fit: 1.0
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 3, 4, 5, 6, 7]

### business_explanation — baseline_rag (target role: business)

- Overall score: **0.9375**
- Concept coverage: 1.0
- Role fit: 0.75
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 4, 5, 6, 7, 8]

### business_explanation — role_aware_agent (target role: business)

- Overall score: **0.9688**
- Concept coverage: 1.0
- Role fit: 0.875
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 4, 5, 6, 7, 8]

### guardrail_deployment — baseline_rag (target role: business)

- Overall score: **0.525**
- Concept coverage: 0.25
- Role fit: 0.0
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 4, 6, 7, 8]

### guardrail_deployment — role_aware_agent (target role: business)

- Overall score: **0.7563**
- Concept coverage: 0.5
- Role fit: 0.625
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 2, 4, 6, 7, 8]

### guardrail_roi — baseline_rag (target role: business)

- Overall score: **0.7837**
- Concept coverage: 0.8
- Role fit: 0.375
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 3, 4, 5, 6, 7, 8]

### guardrail_roi — role_aware_agent (target role: business)

- Overall score: **0.9087**
- Concept coverage: 0.8
- Role fit: 0.875
- Source isolation: 1.0
- Citation support: 1.0
- Hallucination safety: 1.0
- Citation files: ['Documentation - Casual Impact + Transformer - Han.pdf']
- Citation pages: [1, 3, 4, 5, 6, 7, 8]

## Interpretation

This evaluation is a lightweight automated proxy. It checks retrieval source isolation, citation support, role-specific language, concept coverage, and hallucination-risk patterns. The baseline comparison tests whether the expression layer improves role fit while maintaining the same source-grounding behavior. Final grading should still include human review of whether each claim is actually supported by the cited evidence.
