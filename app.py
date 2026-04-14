import os
import json
import tempfile

import streamlit as st
from openai import OpenAI
from rag_system import initialize_rag, load_pdf, load_docx


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="TechMPower RAG Assistant", layout="wide")

st.title("TechMPower RAG Assistant")
st.caption("Document-grounded RAG system with multi-role responses")


# -----------------------------
# Helpers
# -----------------------------
def load_resume_text(uploaded_file) -> str:
    """Read uploaded PDF/DOCX resume and return extracted text."""
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
        return text[:5000]  # keep prompt size reasonable
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def extract_profile_from_resume(resume_text: str):
    """
    Use OpenAI to infer a structured user profile from resume text.
    Returns dict or None.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    if not resume_text or not resume_text.strip():
        return None

    client = OpenAI(api_key=api_key)

    prompt = f"""
Extract a structured user profile from the following resume text.

Return ONLY valid JSON with exactly these keys:
- role: one of ["general", "pm", "engineer", "business"]
- technical_level: one of ["low", "medium", "high"]
- goal: one of ["understanding", "decision", "implementation"]
- short_reason: short explanation of why you inferred this profile

Resume text:
{resume_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract structured user profiles from resumes. "
                    "Return only valid JSON and nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content.strip()

    try:
        profile = json.loads(content)

        valid_roles = {"general", "pm", "engineer", "business"}
        valid_levels = {"low", "medium", "high"}
        valid_goals = {"understanding", "decision", "implementation"}

        if profile.get("role") not in valid_roles:
            profile["role"] = "general"
        if profile.get("technical_level") not in valid_levels:
            profile["technical_level"] = "medium"
        if profile.get("goal") not in valid_goals:
            profile["goal"] = "understanding"
        if "short_reason" not in profile:
            profile["short_reason"] = "No reason provided."

        return profile
    except Exception:
        return None


def display_citations(citations):
    if citations:
        for c in citations:
            page = f"p.{c['page']}" if c["page"] else "doc"
            st.write(
                f"{c['source_file']} | {page} | {c['section']} | {c['aim']} | score={c['score']}"
            )
    else:
        st.write("No citations available.")


# -----------------------------
# Initialize RAG
# -----------------------------
if "rag" not in st.session_state:
    with st.spinner("Loading RAG system..."):
        st.session_state.rag = initialize_rag(docs_dir=".", force_rebuild=False)

rag = st.session_state.rag


# -----------------------------
# Sidebar settings
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

uploaded_file = st.sidebar.file_uploader(
    "Upload resume (PDF/DOCX)",
    type=["pdf", "docx"]
)

use_resume_profile = st.sidebar.checkbox(
    "Use uploaded resume to infer profile",
    value=True
)

allow_manual_override = st.sidebar.checkbox(
    "Allow manual role override",
    value=True
)


# -----------------------------
# Main input
# -----------------------------
query = st.text_area("Enter your question", height=140)


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

            # Resume -> inferred profile
            if uploaded_file is not None and use_resume_profile:
                with st.spinner("Reading resume and inferring user profile..."):
                    resume_text = load_resume_text(uploaded_file)
                    inferred_profile = extract_profile_from_resume(resume_text)

                if inferred_profile and not allow_manual_override:
                    effective_role = inferred_profile["role"]
                elif inferred_profile and allow_manual_override:
                    # Manual role stays in control when override is allowed
                    effective_role = manual_role

            # Show profile information
            st.subheader("Active Profile")
            if inferred_profile:
                st.json(
                    {
                        "inferred_profile": inferred_profile,
                        "effective_role_used_for_generation": effective_role,
                    }
                )
            else:
                st.json(
                    {
                        "effective_role_used_for_generation": effective_role,
                        "profile_source": "manual selection",
                    }
                )

            # Generate answer
            with st.spinner("Generating answer..."):
                result = rag.answer_question(
                    query=query,
                    mode=mode,
                    role=effective_role
                )

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Top Citations")
            display_citations(result["citations"])

            if show_context:
                st.subheader("Retrieved Context")
                st.text(result["retrieved_context"])

        except Exception as e:
            st.error(f"Error: {e}")
