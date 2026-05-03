"""
evaluate_project_agent.py

Evaluate the Background-Aware Project Explanation Agent.

This script tests:
1. Retrieval correctness
2. Source isolation
3. Grounding / hallucination risk
4. Role adaptation
5. Completeness by role
6. Citation coverage

Run:
    python evaluate_project_agent.py --docs_dir data/sample_docs/uploaded_projects/demo_user --output_dir eval_outputs

Before running:
    export OPENAI_API_KEY="your_key"
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any

from retrieval.rag_system import initialize_rag


# -----------------------------
# Test set
# -----------------------------

TEST_CASES = [
    {
        "id": "general_summary",
        "question": "Summarize the uploaded project document and identify the main project or projects inside it.",
        "role": "general",
        "mode": "summary",
        "expected_concepts": [
            "project",
            "objective",
            "workflow",
            "data",
            "model",
            "result",
            "limitation",
        ],
    },
    {
        "id": "engineer_explanation",
        "question": "Explain this project for an engineer.",
        "role": "engineer",
        "mode": "qa",
        "expected_concepts": [
            "workflow",
            "pipeline",
            "data",
            "feature",
            "model",
            "metric",
            "evaluation",
            "risk",
            "next step",
        ],
    },
    {
        "id": "engineer_workflow",
        "question": "What is the end-to-end workflow of this project?",
        "role": "engineer",
        "mode": "qa",
        "expected_concepts": [
            "input",
            "workflow",
            "pipeline",
            "data",
            "feature",
            "model",
            "output",
        ],
    },
    {
        "id": "engineer_model_roles",
        "question": "What models, modules, or system components serve different roles in this project?",
        "role": "engineer",
        "mode": "qa",
        "expected_concepts": [
            "model",
            "module",
            "role",
            "workflow",
            "evaluation",
        ],
    },
    {
        "id": "engineer_evaluation_results",
        "question": "What are the evaluation results and what do they imply?",
        "role": "engineer",
        "mode": "qa",
        "expected_concepts": [
            "result",
            "metric",
            "R²",
            "RMSE",
            "MAE",
            "evaluation",
            "performance",
            "weak",
            "validation",
        ],
    },
    {
        "id": "pm_explanation",
        "question": "Explain this project for a product manager.",
        "role": "pm",
        "mode": "qa",
        "expected_concepts": [
            "user",
            "goal",
            "workflow",
            "input",
            "output",
            "risk",
            "success criteria",
            "next step",
        ],
    },
    {
        "id": "business_explanation",
        "question": "Explain this project for a business owner. Distinguish documented outcomes from potential value.",
        "role": "business",
        "mode": "summary",
        "expected_concepts": [
            "business",
            "value",
            "decision",
            "documented",
            "potential",
            "risk",
            "next step",
        ],
    },
    {
        "id": "guardrail_deployment",
        "question": "Was this project deployed in production? Answer only from the uploaded document.",
        "role": "business",
        "mode": "qa",
        "expected_concepts": [
            "not specified",
            "retrieved evidence",
            "production",
            "deployment",
        ],
    },
    {
        "id": "guardrail_roi",
        "question": "Did this project prove positive ROI, investment return, or cost savings? Answer only from the uploaded document.",
        "role": "business",
        "mode": "qa",
        "expected_concepts": [
            "not specified",
            "retrieved evidence",
            "ROI",
            "cost",
            "return",
        ],
    },
]


ROLE_KEYWORDS = {
    "engineer": [
        "workflow",
        "pipeline",
        "architecture",
        "data",
        "feature",
        "model",
        "metric",
        "validation",
        "risk",
        "implementation",
    ],
    "pm": [
        "user",
        "workflow",
        "goal",
        "output",
        "risk",
        "success",
        "next step",
        "product",
    ],
    "business": [
        "business",
        "value",
        "decision",
        "documented",
        "potential",
        "risk",
        "leadership",
        "stakeholder",
    ],
    "general": [
        "project",
        "goal",
        "data",
        "method",
        "result",
        "limitation",
    ],
}


UNSUPPORTED_RISK_PATTERNS = [
    r"\bROI\b",
    r"\bcost savings\b",
    r"\brevenue increase\b",
    r"\bdeployed in production\b",
    r"\bproduction-ready\b",
    r"\bproven business impact\b",
    r"\bguaranteed\b",
    r"\bwill improve\b",
    r"\bcaused\b",
    r"\bdefinitively proves\b",
]


SAFE_UNCERTAINTY_PHRASES = [
    "not specified",
    "retrieved evidence does not",
    "does not show",
    "does not provide",
    "not documented",
    "if validated",
    "could support",
    "potential",
]


# -----------------------------
# Scoring helpers
# -----------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def concept_coverage_score(answer: str, expected_concepts: List[str]) -> float:
    """
    Lightweight completeness score.
    Counts how many expected concepts appear in the answer.
    """
    if not expected_concepts:
        return 0.0

    low = normalize_text(answer)
    hits = 0

    for concept in expected_concepts:
        c = concept.lower()
        if c in low:
            hits += 1

    return hits / len(expected_concepts)


def role_fit_score(answer: str, role: str) -> float:
    """
    Lightweight role-fit score based on role-specific vocabulary.
    This is not a perfect metric, but useful for comparing runs.
    """
    keywords = ROLE_KEYWORDS.get(role, ROLE_KEYWORDS["general"])
    return concept_coverage_score(answer, keywords)


def source_isolation_score(citations: List[Dict[str, Any]], allowed_files: List[str]) -> float:
    """
    Checks whether citations only come from the currently uploaded project docs.
    """
    if not citations:
        return 0.0

    allowed = set(allowed_files)
    cited_files = [c.get("source_file") for c in citations]

    if not cited_files:
        return 0.0

    valid = sum(1 for f in cited_files if f in allowed)
    return valid / len(cited_files)


def citation_count_score(citations: List[Dict[str, Any]]) -> float:
    """
    Rewards having some citations, capped at 1.0.
    """
    return min(len(citations) / 5.0, 1.0)


def hallucination_risk_score(answer: str) -> float:
    """
    Higher is better.
    Penalizes risky unsupported-sounding claims, but does not penalize cases
    where the answer clearly says those risky claims are not documented/proven.
    """
    low = normalize_text(answer)

    negation_patterns = [
        r"no evidence.{0,100}(roi|cost savings|investment return|production|deployed|business impact)",
        r"does not.{0,100}(prove|show|provide|document|include).{0,100}(roi|cost savings|investment return|production|deployment|business impact)",
        r"not.{0,100}(deployed|production-ready|proven|documented|specified)",
        r"(roi|cost savings|investment return|production|deployment).{0,100}(not proven|not documented|not specified)",
    ]

    if any(re.search(pattern, low) for pattern in negation_patterns):
        return 1.0

    risk_hits = 0
    for pattern in UNSUPPORTED_RISK_PATTERNS:
        if re.search(pattern.lower(), low):
            risk_hits += 1

    has_uncertainty = any(phrase in low for phrase in SAFE_UNCERTAINTY_PHRASES)

    if risk_hits == 0:
        return 1.0

    if has_uncertainty:
        return max(0.7, 1.0 - 0.1 * risk_hits)

    return max(0.0, 1.0 - 0.3 * risk_hits)


def page_coverage(citations: List[Dict[str, Any]]) -> List[Any]:
    pages = []
    for c in citations:
        page = c.get("page")
        if page is not None and page not in pages:
            pages.append(page)
    return pages


def compute_overall_score(row: Dict[str, Any]) -> float:
    """
    Weighted score:
    - completeness / concept coverage
    - role fit
    - source isolation
    - citation support
    - hallucination risk
    """
    weights = {
        "concept_coverage": 0.30,
        "role_fit": 0.25,
        "source_isolation": 0.20,
        "citation_support": 0.10,
        "hallucination_safety": 0.15,
    }

    return round(
        row["concept_coverage"] * weights["concept_coverage"]
        + row["role_fit"] * weights["role_fit"]
        + row["source_isolation"] * weights["source_isolation"]
        + row["citation_support"] * weights["citation_support"]
        + row["hallucination_safety"] * weights["hallucination_safety"],
        4,
    )


# -----------------------------
# Evaluation runner
# -----------------------------

def evaluate(docs_dir: str, output_dir: str, force_rebuild: bool = True) -> List[Dict[str, Any]]:
    docs_path = Path(docs_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    allowed_files = sorted(
        [
            p.name
            for p in docs_path.iterdir()
            if p.suffix.lower() in {".pdf", ".docx", ".md", ".txt"}
        ]
    )

    if not allowed_files:
        raise FileNotFoundError(f"No project documents found in {docs_dir}")

    print("=" * 90)
    print("Allowed project files:")
    for file in allowed_files:
        print(f"- {file}")
    print("=" * 90)

    rag = initialize_rag(docs_dir=docs_dir, force_rebuild=force_rebuild)

    rows: List[Dict[str, Any]] = []

    for case in TEST_CASES:
        print(f"\nRunning test: {case['id']} | target_role={case['role']}")

        systems_to_test = [
            {
                "system_type": "baseline_rag",
                "call_role": "general",
                "apply_expression_layer": False,
            },
            {
                "system_type": "role_aware_agent",
                "call_role": case["role"],
                "apply_expression_layer": True,
            },
        ]

        for system_cfg in systems_to_test:
            print(f"  System: {system_cfg['system_type']}")

            try:
                result = rag.answer_question(
                    query=case["question"],
                    mode=case["mode"],
                    role=system_cfg["call_role"],
                    apply_expression_layer=system_cfg["apply_expression_layer"],
                )

                answer = result.get("answer", "")
                citations = result.get("citations", [])

                row = {
                    "test_id": case["id"],
                    "system_type": system_cfg["system_type"],
                    "question": case["question"],
                    "target_role": case["role"],
                    "call_role": system_cfg["call_role"],
                    "mode": case["mode"],
                    "answer": answer,
                    "num_citations": len(citations),
                    "citation_files": sorted(list({c.get("source_file") for c in citations})),
                    "citation_pages": page_coverage(citations),
                    "concept_coverage": round(concept_coverage_score(answer, case["expected_concepts"]), 4),
                    "role_fit": round(role_fit_score(answer, case["role"]), 4),
                    "source_isolation": round(source_isolation_score(citations, allowed_files), 4),
                    "citation_support": round(citation_count_score(citations), 4),
                    "hallucination_safety": round(hallucination_risk_score(answer), 4),
                    "error": "",
                }
                row["overall_score"] = compute_overall_score(row)

            except Exception as e:
                row = {
                    "test_id": case["id"],
                    "system_type": system_cfg["system_type"],
                    "question": case["question"],
                    "target_role": case["role"],
                    "call_role": system_cfg["call_role"],
                    "mode": case["mode"],
                    "answer": "",
                    "num_citations": 0,
                    "citation_files": [],
                    "citation_pages": [],
                    "concept_coverage": 0.0,
                    "role_fit": 0.0,
                    "source_isolation": 0.0,
                    "citation_support": 0.0,
                    "hallucination_safety": 0.0,
                    "overall_score": 0.0,
                    "error": str(e),
                }

            rows.append(row)

            print(f"  Overall score: {row['overall_score']}")
            if row["error"]:
                print(f"  ERROR: {row['error']}")
            else:
                print(f"  Citation files: {row['citation_files']}")
                print(f"  Citation pages: {row['citation_pages']}")

    # Save JSON
    json_path = output_path / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # Save CSV summary
    csv_path = output_path / "evaluation_summary.csv"
    csv_fields = [
        "test_id",
        "system_type",
        "target_role",
        "call_role",
        "mode",
        "overall_score",
        "concept_coverage",
        "role_fit",
        "source_isolation",
        "citation_support",
        "hallucination_safety",
        "num_citations",
        "citation_files",
        "citation_pages",
        "error",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in csv_fields})

    def avg(rows_subset: List[Dict[str, Any]], key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in rows_subset) / len(rows_subset) if rows_subset else 0.0

    baseline_rows = [r for r in rows if r.get("system_type") == "baseline_rag"]
    agent_rows = [r for r in rows if r.get("system_type") == "role_aware_agent"]

    avg_score = avg(rows, "overall_score")
    avg_grounding = avg(rows, "source_isolation")
    avg_role_fit = avg(rows, "role_fit")
    avg_hallucination_safety = avg(rows, "hallucination_safety")

    baseline_overall = avg(baseline_rows, "overall_score")
    agent_overall = avg(agent_rows, "overall_score")
    baseline_role_fit = avg(baseline_rows, "role_fit")
    agent_role_fit = avg(agent_rows, "role_fit")
    baseline_source_isolation = avg(baseline_rows, "source_isolation")
    agent_source_isolation = avg(agent_rows, "source_isolation")
    baseline_hallucination_safety = avg(baseline_rows, "hallucination_safety")
    agent_hallucination_safety = avg(agent_rows, "hallucination_safety")

    # Save markdown report
    md_path = output_path / "evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Project Agent Evaluation Report\n\n")
        f.write("## Evaluation setup\n\n")
        f.write(f"- Docs directory: `{docs_dir}`\n")
        f.write(f"- Allowed project files: {', '.join(allowed_files)}\n")
        f.write(f"- Number of test cases: {len(TEST_CASES)}\n")
        f.write("- Systems compared: baseline RAG vs role-aware agent\n\n")

        f.write("## Aggregate scores\n\n")
        f.write(f"- Average overall score across all runs: **{avg_score:.3f}**\n")
        f.write(f"- Average source isolation across all runs: **{avg_grounding:.3f}**\n")
        f.write(f"- Average role fit across all runs: **{avg_role_fit:.3f}**\n")
        f.write(f"- Average hallucination safety across all runs: **{avg_hallucination_safety:.3f}**\n\n")

        f.write("## Baseline comparison\n\n")
        f.write("| System | Overall score | Role fit | Source isolation | Hallucination safety |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        f.write(f"| Baseline RAG | {baseline_overall:.3f} | {baseline_role_fit:.3f} | {baseline_source_isolation:.3f} | {baseline_hallucination_safety:.3f} |\n")
        f.write(f"| Role-aware agent | {agent_overall:.3f} | {agent_role_fit:.3f} | {agent_source_isolation:.3f} | {agent_hallucination_safety:.3f} |\n\n")
        f.write(
            "The baseline uses the same retrieval system but disables the expression layer. "
            "The role-aware agent uses the full expression layer to adapt the same retrieved evidence "
            "for the target audience while preserving document grounding.\n\n"
        )

        f.write("## Test results\n\n")
        for row in rows:
            f.write(f"### {row['test_id']} — {row['system_type']} (target role: {row['target_role']})\n\n")
            f.write(f"- Overall score: **{row['overall_score']}**\n")
            f.write(f"- Concept coverage: {row['concept_coverage']}\n")
            f.write(f"- Role fit: {row['role_fit']}\n")
            f.write(f"- Source isolation: {row['source_isolation']}\n")
            f.write(f"- Citation support: {row['citation_support']}\n")
            f.write(f"- Hallucination safety: {row['hallucination_safety']}\n")
            f.write(f"- Citation files: {row['citation_files']}\n")
            f.write(f"- Citation pages: {row['citation_pages']}\n")
            if row["error"]:
                f.write(f"- Error: `{row['error']}`\n")
            f.write("\n")

        f.write("## Interpretation\n\n")
        f.write(
            "This evaluation is a lightweight automated proxy. It checks retrieval source isolation, "
            "citation support, role-specific language, concept coverage, and hallucination-risk patterns. "
            "The baseline comparison tests whether the expression layer improves role fit while maintaining "
            "the same source-grounding behavior. Final grading should still include human review of whether "
            "each claim is actually supported by the cited evidence.\n"
        )

    print("\n" + "=" * 90)
    print("Evaluation complete.")
    print(f"JSON results: {json_path}")
    print(f"CSV summary: {csv_path}")
    print(f"Markdown report: {md_path}")
    print("=" * 90)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs_dir",
        required=True,
        help="Directory containing the currently uploaded project documents.",
    )
    parser.add_argument(
        "--output_dir",
        default="eval_outputs",
        help="Directory to save evaluation outputs.",
    )
    parser.add_argument(
        "--no_force_rebuild",
        action="store_true",
        help="Use existing FAISS index instead of rebuilding.",
    )

    args = parser.parse_args()

    evaluate(
        docs_dir=args.docs_dir,
        output_dir=args.output_dir,
        force_rebuild=not args.no_force_rebuild,
    )


if __name__ == "__main__":
    main()