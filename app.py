import os
import tempfile
import math
import shutil
from pathlib import Path

from dotenv import load_dotenv

# Load project-level environment variables before initializing any API clients.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

import streamlit as st
from openai import OpenAI

from retrieval.rag_system import initialize_rag, load_pdf, load_docx
from memory.background_memory import onboard_user_background, retrieve_user_background
from core.query_orchestrator import process_query
from core.expression_layer import generate_personalized_explanation


st.set_page_config(page_title="TechMPower RAG Assistant", layout="wide")

st.title("TechMPower RAG Assistant")
st.caption("Document-grounded RAG system with background-aware personalization")


# -----------------------------
# Helpers
# -----------------------------
def load_resume_text(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            pages = load_pdf(tmp_path)
        elif suffix == ".docx":
            pages = load_docx(tmp_path)
        else:
            return ""

        text = " ".join(page_text for _, page_text in pages)
        return text[:8000]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def save_uploaded_project_docs(uploaded_files, user_id: str) -> tuple[str | None, list[str]]:
    """Save uploaded project documents and return the active docs directory.

    Resume uploads are used for background personalization.
    Project-document uploads are used as the active RAG knowledge base.
    """
    if not uploaded_files:
        return None, []

    safe_user_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in user_id)
    upload_dir = PROJECT_ROOT / "data" / "sample_docs" / "uploaded_projects" / safe_user_id

    # Each upload action should define the active project knowledge base.
    # Clear previous project files for this user so a new upload does not mix
    # with older uploaded PDFs/MD files in the same folder.
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name).name
        file_path = upload_dir / filename
        file_path.write_bytes(uploaded_file.getbuffer())
        saved_files.append(str(file_path))

    return str(upload_dir), saved_files


def _format_citation_score(score) -> str:
    """Format citation score safely and hide NaN values from the UI."""
    if score is None:
        return ""
    try:
        score_float = float(score)
        if math.isnan(score_float):
            return ""
        return f" · relevance={score_float:.3f}"
    except (TypeError, ValueError):
        return ""


def display_citations(citations):
    """Display citations in a readable project-report style.

    Avoid exposing raw implementation fields such as score=nan because they make
    the app look broken even when retrieval returned valid document pages.
    """
    if not citations:
        st.write("No citations available. Try asking a more specific question, such as the project name, model name, or metric you want explained.")
        return

    for idx, c in enumerate(citations, start=1):
        source = c.get("source_file", "unknown source")
        page = f"p.{c.get('page')}" if c.get("page") else "document"
        section = c.get("section") or "section unknown"
        aim = c.get("aim") or "evidence"
        score_text = _format_citation_score(c.get("score"))

        st.markdown(
            f"**[{idx}] {source}, {page}**  \n"
            f"Section: `{section}` · Evidence type: `{aim}`{score_text}"
        )



def build_user_profile_from_background(retrieved_background: dict) -> dict:
    structured = (retrieved_background or {}).get("structured_profile") or {}

    role = structured.get("role_lens", "general")
    if role == "product_manager":
        role = "pm"

    return {
        "role": role,
        "technical_level": structured.get("technical_depth", "medium"),
        "goal": "understanding",
        "short_reason": structured.get("short_reason", "")
    }


# ----------------------------------------------------
# Stronger project-summary query adapter for project docs
# ----------------------------------------------------
def build_project_aware_query(raw_query: str, mode: str, role: str, has_project_docs: bool) -> str:
    """Strengthen the user query when uploaded project docs are available.

    This prevents the app from producing generic component-style summaries and pushes
    the RAG layer to extract concrete project details from the uploaded document.
    """
    if not has_project_docs:
        return raw_query

    query_lower = raw_query.lower()
    project_summary_triggers = [
        "summarize", "summary", "explain", "overview", "what is this project",
        "what does this project do", "project about", "describe this project",
        "tell me about this project", "walk me through this project"
    ]

    if mode != "summary" and not any(trigger in query_lower for trigger in project_summary_triggers):
        return raw_query

    return f"""
The user uploaded one or more project documents and wants a grounded project summary.

Original user question:
{raw_query}

Summarize the uploaded document as a real project, not as a generic software architecture template.

Instructions:
1. Treat only the currently uploaded project document(s) as the factual project evidence. Do not use resume/background memory or previous uploads as project facts.
2. First identify whether the uploaded document contains one project or multiple projects/assignments.
3. If it contains multiple projects, give a short inventory of the main projects first, then summarize the most relevant one based on the user question.
4. For the selected project, include:
   - project title / domain
   - actual artifact type, such as report, README, portfolio website, dashboard, notebook, or model
   - business, analytical, or technical objective
   - users / stakeholders when specified
   - dataset and target variable if it is a modeling project
   - data preparation steps if specified
   - modeling methods used if specified
   - model selection criteria and selected model only if explicitly supported
   - key quantitative results only if explicitly supported
   - system / implementation details when it is a software, website, dashboard, or RAG project
   - limitations / risks
   - next steps
5. Use concrete details from the uploaded document, including tech stack, files, modules, deployment steps, visualizations, model names, metrics, variables, and page-level evidence when available.
6. Do not invent missing ML details for non-ML artifacts. If the uploaded document is a README or website project, explain the architecture, file structure, UI features, deployment, customization workflow, and engineering risks instead of forcing dataset/model sections.
7. Do not output generic module names like Data Acquisition Module, Model Construction Module, or Expert Consultation Module unless the uploaded document explicitly uses those terms.
8. Adapt the explanation for the selected response perspective: {role}.
""".strip()


# ----------------------------------------------------
# Compact, content-focused retrieval query for project docs
# ----------------------------------------------------
def build_project_retrieval_query(raw_query: str, mode: str, has_project_docs: bool) -> str:
    """Build a compact, document-agnostic retrieval query for uploaded project docs.

    Do not hard-code project names. The query should work for any uploaded PDF,
    report, resume project, README, case study, or portfolio document.
    """
    if not has_project_docs:
        return raw_query

    query_lower = raw_query.lower()
    retrieval_hints = []

    # Generic project-summary retrieval.
    if mode == "summary" or any(term in query_lower for term in ["summarize", "summary", "explain", "overview", "project"]):
        retrieval_hints.append(
            "project title domain objective problem statement overview features tech stack architecture file structure implementation deployment customization visualizations PDF reports dataset data source target variable methodology data preparation feature engineering model development results conclusion limitations future work"
        )

    # Generic model-selection retrieval.
    if any(term in query_lower for term in ["selected", "selection", "best model", "chosen", "why", "deploy", "recommend"]):
        retrieval_hints.append(
            "model selection selected model chosen model best model selection criteria comparison table metrics AIC BIC SBC RMSE MAE accuracy ROC AUC log loss R-squared conclusion caveats limitations"
        )

    # Generic evaluation/results retrieval.
    if any(term in query_lower for term in ["result", "metric", "performance", "evaluate", "evaluation", "accuracy", "rmse", "mae", "auc", "sbc", "aic", "bic"]):
        retrieval_hints.append(
            "performance results evaluation metrics validation summary table quantitative results model comparison"
        )

    # Generic data-preparation retrieval.
    if any(term in query_lower for term in ["data", "dataset", "clean", "preparation", "imputation", "missing", "outlier", "feature"]):
        retrieval_hints.append(
            "data exploration data preparation missing values imputation outliers transformations feature creation derived variables"
        )

    # Generic software / README / portfolio retrieval.
    if any(term in query_lower for term in ["website", "portfolio", "readme", "app", "application", "frontend", "deployment", "deploy", "architecture", "file structure", "tech stack", "implementation", "engineer"]):
        retrieval_hints.append(
            "features tech stack HTML CSS JavaScript Python architecture file structure assets images pdfs deployment GitHub Pages Netlify Vercel customization contact form performance responsive design implementation engineering risks"
        )

    # Generic limitations / next-step retrieval.
    if any(term in query_lower for term in ["limitation", "risk", "future", "next step", "improve", "weakness"]):
        retrieval_hints.append(
            "limitations risks caveats future work next steps improvement recommendations monitoring"
        )

    if retrieval_hints:
        # Keep the original user wording first so named entities from any uploaded document remain searchable.
        return raw_query + "\n\nFocus retrieval on: " + "; ".join(dict.fromkeys(retrieval_hints))

    return raw_query


def answer_with_external_knowledge(
    query: str,
    user_profile: dict | None = None,
    role: str = "general"
) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    technical_level = "medium"
    goal = "understanding"
    short_reason = ""

    if user_profile:
        technical_level = user_profile.get("technical_level", "medium")
        goal = user_profile.get("goal", "understanding")
        short_reason = user_profile.get("short_reason", "")

    prompt = f"""
You are a helpful assistant.

The user is asking a general concept question, not a question tied to the uploaded project PDF.
Generate a neutral, accurate base explanation first. Do not personalize heavily here.
The explicit expression layer will personalize the answer later.

Instructions:
- Answer clearly and accurately using general knowledge.
- Focus on factual correctness and conceptual coverage.
- Do not over-adapt to the user's role in this step.
- Do not say "the evidence is insufficient."
- Do not say "human review required."
- Do not mention missing project documents.

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {
                "role": "system",
                "content": "You explain concepts clearly and adapt explanations to the user's background."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
    )

    base_explanation = response.choices[0].message.content.strip()

    query_understanding = {
        "query_type": "concept_explanation",
        "topic": query.strip().rstrip("?")[:120] if query else "unknown",
        "intent": "understand_concept",
        "domain": "general knowledge",
        "requires_background_retrieval": True,
        "requires_project_context": False,
        "needs_clarification": False,
    }

    expression_background_package = {
        "structured_profile": {
            "role_lens": role,
            "technical_depth": technical_level,
            "technical_level": technical_level,
            "jargon_tolerance": (user_profile or {}).get("jargon_tolerance", "medium"),
            "preferred_explanation_style": (user_profile or {}).get("preferred_explanation_style", []),
            "goal": goal,
            "short_reason": short_reason,
        },
        "retrieved_background_chunks": [
            {
                "chunk_type": "profile_inference",
                "text": short_reason,
            }
        ] if short_reason else []
    }

    expression_result = generate_personalized_explanation(
        base_explanation=base_explanation,
        query_understanding=query_understanding,
        retrieved_background_package=expression_background_package,
        role=role,
    )

    return {
        "answer": expression_result["final_explanation"],
        "base_explanation": expression_result["base_explanation"],
        "expression_plan": expression_result["expression_plan"],
        "quality_report": expression_result.get("quality_report", {}),
        "query_understanding": expression_result.get("query_understanding", query_understanding),
        "retrieved_background_package": expression_result.get("retrieved_background_package", expression_background_package),
        "citations": [],
        "retrieved_context": "External/general knowledge route"
    }


# -----------------------------
# Init RAG
# -----------------------------
DEFAULT_DOCS_DIR = str(PROJECT_ROOT / "data" / "sample_docs")

if "active_docs_dir" not in st.session_state:
    st.session_state.active_docs_dir = DEFAULT_DOCS_DIR

if "rag" not in st.session_state:
    with st.spinner("Loading default project knowledge base..."):
        st.session_state.rag = initialize_rag(
            docs_dir=st.session_state.active_docs_dir,
            force_rebuild=False
        )

rag = st.session_state.rag


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

mode = st.sidebar.selectbox(
    "Choose mode",
    ["qa", "summary", "coding"]
)

manual_role = st.sidebar.selectbox(
    "Choose response perspective",
    ["general", "pm", "engineer", "business"]
)

show_context = st.sidebar.checkbox("Show retrieved context", value=False)
show_debug = st.sidebar.checkbox("Show debug info", value=False)

uploaded_file = st.sidebar.file_uploader(
    "Upload resume (PDF/DOCX)",
    type=["pdf", "docx"]
)

project_files = st.sidebar.file_uploader(
    "Upload project docs for RAG (PDF/DOCX/MD/TXT)",
    type=["pdf", "docx", "md", "txt"],
    accept_multiple_files=True,
    help="Upload project reports, README files, datasheets, or portfolio docs. These files become the active project knowledge base for grounded answers."
)

use_resume_profile = st.sidebar.checkbox(
    "Use uploaded resume to infer profile",
    value=True
)

allow_manual_override = st.sidebar.checkbox(
    "Allow manual role override",
    value=True
)


user_id = st.sidebar.text_input("User ID", value="demo_user")

# -----------------------------
# Project document upload / active RAG source
# -----------------------------
has_project_docs = bool(project_files) or st.session_state.get("project_upload_signature") is not None

if project_files:
    uploaded_docs_dir, saved_project_files = save_uploaded_project_docs(project_files, user_id=user_id)
    upload_signature = "|".join(sorted(saved_project_files))

    if st.session_state.get("project_upload_signature") != upload_signature:
        with st.spinner("Indexing uploaded project documents..."):
            st.session_state.rag = initialize_rag(
                docs_dir=uploaded_docs_dir,
                force_rebuild=True
            )
            st.session_state.active_docs_dir = uploaded_docs_dir
            st.session_state.project_upload_signature = upload_signature

    rag = st.session_state.rag
    st.sidebar.success(f"Using {len(saved_project_files)} uploaded project document(s) for RAG.")
else:
    rag = st.session_state.rag
    if st.session_state.get("project_upload_signature") is not None:
        st.sidebar.success("Using previously uploaded project document(s) for RAG.")
    else:
        st.sidebar.info("Using default sample_docs knowledge base for RAG.")


# -----------------------------
# Main input
# -----------------------------
query = st.text_area("Enter your question", height=140)

st.caption(
    "Tip: Upload a resume to personalize the answer, and upload project docs to ground the answer in your project. "
    "Then switch the response perspective to general, PM, engineer, or business."
)


# -----------------------------
# Run
# -----------------------------
if st.button("Run"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            inferred_profile = None
            effective_role = manual_role
            retrieved_background = None
            orchestration_result = None
            query_understanding = None
            routing_decision = None

            # -----------------------------
            # Step 1: Background onboarding
            # -----------------------------
            if uploaded_file is not None and use_resume_profile:
                with st.spinner("Reading resume and onboarding user background..."):
                    resume_text = load_resume_text(uploaded_file)

                    onboard_user_background(
                        user_id=user_id,
                        raw_background_inputs=[
                            {
                                "source_type": "resume",
                                "raw_text": resume_text
                            }
                        ]
                    )

            # -----------------------------
            # Step 2: Query understanding + routing
            # -----------------------------
            # Run query understanding silently so the user-facing app stays focused on the answer.
            orchestration_result = process_query(
                user_id=user_id,
                raw_query=query,
                has_uploaded_project_doc=has_project_docs
            )

            query_understanding = orchestration_result["query_understanding_object"]
            routing_decision = orchestration_result["routing_decision"]

            # -----------------------------
            # Step 3: Clarification route
            # -----------------------------
            if routing_decision["route"] == "clarification":
                st.info(routing_decision["message"])

                if show_debug:
                    st.markdown("## Debug Panel")

                    with st.expander("Query Understanding", expanded=False):
                        st.json(query_understanding)

                    with st.expander("Routing Decision", expanded=False):
                        st.json(routing_decision)

                    orchestration_metadata = (orchestration_result or {}).get("orchestration_metadata", {})
                    if orchestration_metadata:
                        with st.expander("Orchestration Metadata", expanded=False):
                            st.json(orchestration_metadata)

            # -----------------------------
            # Step 4: Retrieval + generation routes
            # -----------------------------
            else:
                if "background_request" in routing_decision:
                    bg_req = routing_decision["background_request"]

                    retrieved_background = retrieve_user_background(
                        user_id=bg_req["user_id"],
                        query=bg_req["query"],
                        recommended_chunk_types=bg_req["recommended_background_chunk_types"]
                    )

                    if retrieved_background.get("structured_profile") is not None:
                        inferred_profile = build_user_profile_from_background(retrieved_background)

                # -----------------------------
                # Role selection
                # -----------------------------
                # If manual override is enabled, always respect the user's dropdown choice.
                # If it is disabled, allow the inferred resume/profile role to drive the answer.
                if allow_manual_override:
                    effective_role = manual_role
                elif inferred_profile:
                    effective_role = inferred_profile["role"]
                else:
                    effective_role = manual_role

                # -----------------------------
                # Keep resume/background separate from project factual grounding
                # -----------------------------
                # Resume/background memory may adapt tone, depth, and role perspective.
                # It must not be used as factual evidence about the uploaded project docs.
                retrieved_background_for_expression = retrieved_background
                if has_project_docs and retrieved_background:
                    retrieved_background_for_expression = {
                        "structured_profile": retrieved_background.get("structured_profile"),
                        "retrieved_background_chunks": [],
                        "usage_policy": "personalization_only_not_project_evidence",
                    }

                # Uploaded project docs should force the expression layer into project-document mode.
                if has_project_docs and query_understanding:
                    query_understanding = dict(query_understanding)
                    query_understanding["query_type"] = "project_explanation"
                    query_understanding["intent"] = "summarize_project_evidence"
                    query_understanding["domain"] = "uploaded project documents"
                    query_understanding["requires_project_context"] = True

                if show_debug:
                    st.markdown("## Debug Panel")

                    with st.expander("Query Understanding", expanded=False):
                        st.json(query_understanding)

                    with st.expander("Routing Decision", expanded=False):
                        st.json(routing_decision)

                    orchestration_metadata = (orchestration_result or {}).get("orchestration_metadata", {})
                    if orchestration_metadata:
                        with st.expander("Orchestration Metadata", expanded=False):
                            st.json(orchestration_metadata)

                # -----------------------------
                # Step 5: Generate answer
                # -----------------------------
                with st.spinner("Generating answer..."):
                    route = routing_decision["route"]

                    if route == "external_knowledge_then_expression" and not has_project_docs:
                        result = answer_with_external_knowledge(
                            query=query,
                            user_profile=inferred_profile,
                            role=effective_role
                        )
                    else:
                        result = rag.answer_question(
                            query=build_project_aware_query(
                                raw_query=query,
                                mode=mode,
                                role=effective_role,
                                has_project_docs=has_project_docs,
                            ),
                            retrieval_query=build_project_retrieval_query(
                                raw_query=query,
                                mode=mode,
                                has_project_docs=has_project_docs,
                            ),
                            mode=mode,
                            role=effective_role,
                            user_profile=inferred_profile,
                            query_understanding=query_understanding,
                            retrieved_background_package=retrieved_background_for_expression,
                            apply_expression_layer=True,
                        )

                st.markdown("## Answer")
                st.markdown(result["answer"])

                if show_debug:
                    st.markdown("## Agent Trace")

                    with st.expander("Retrieved Background Package", expanded=False):
                        if retrieved_background:
                            st.write("Original retrieved background memory:")
                            st.json(retrieved_background)
                            if has_project_docs:
                                st.write("Background package passed to expression layer after project-grounding guardrail:")
                                st.json(retrieved_background_for_expression)
                        else:
                            st.write("No background package was retrieved for this route.")
                            if has_project_docs:
                                st.write("Project documents were uploaded and used by the RAG knowledge base, not by background memory.")

                    retrieval_diagnostics = (retrieved_background or {}).get("retrieval_diagnostics")
                    if retrieval_diagnostics:
                        with st.expander("Retrieval Diagnostics", expanded=False):
                            st.json(retrieval_diagnostics)

                    with st.expander("Base Explanation", expanded=False):
                        st.write(result.get("base_explanation", "No base explanation available."))

                    with st.expander("Expression Plan", expanded=False):
                        expression_plan = result.get("expression_plan")
                        if expression_plan:
                            st.json(expression_plan)
                        else:
                            st.write("No expression plan available.")

                    with st.expander("Quality Report", expanded=True):
                        quality_report = result.get("quality_report")
                        if quality_report:
                            st.json(quality_report)
                        else:
                            st.write("No quality report available.")

                    with st.expander("Final Personalized Explanation", expanded=False):
                        st.write(result.get("answer", "No final answer available."))

                if has_project_docs:
                    st.caption("This answer was grounded in the uploaded project document knowledge base and then adapted to the selected response perspective.")
                st.markdown("## Citations")
                display_citations(result.get("citations", []))

                if show_context:
                    with st.expander("Retrieved Context", expanded=False):
                        st.text(result.get("retrieved_context", ""))

        except Exception as e:
            st.error(f"Error: {e}")

