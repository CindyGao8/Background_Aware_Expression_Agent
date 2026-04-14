
import streamlit as st
from rag_system import initialize_rag

st.set_page_config(page_title="TechMPower RAG Assistant", layout="wide")

st.title("TechMPower RAG Assistant")
st.caption("Document-grounded RAG system with multi-role responses")

if "rag" not in st.session_state:
    st.session_state.rag = initialize_rag(docs_dir=".", force_rebuild=True)

rag = st.session_state.rag

st.sidebar.header("Settings")

mode = st.sidebar.selectbox(
    "Choose mode",
    ["qa", "summary", "coding"]
)

role = st.sidebar.selectbox(
    "Choose response perspective",
    ["general", "pm", "engineer", "business"]
)

show_context = st.sidebar.checkbox("Show retrieved context", value=False)

query = st.text_area("Enter your question")

if st.button("Run"):
    if query.strip():
        try:
            result = rag.answer_question(query=query, mode=mode, role=role)

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Top Citations")
            if result["citations"]:
                for c in result["citations"]:
                    page = f"p.{c['page']}" if c["page"] else "doc"
                    st.write(
                        f"{c['source_file']} | {page} | {c['section']} | {c['aim']} | score={c['score']}"
                    )
            else:
                st.write("No citations available.")

            if show_context:
                st.subheader("Retrieved Context")
                st.text(result["retrieved_context"])

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")
