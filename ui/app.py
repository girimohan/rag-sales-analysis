import streamlit as st
import requests

st.set_page_config(
    page_title="Superstore RAG Assistant",
    page_icon="🛒",
    layout="centered",
)

st.title("Superstore RAG Assistant")
st.markdown(
    "Ask any question about sales, customers, products, or regions "
    "and get an answer based on the Superstore dataset."
)

with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox(
        "LLM Model",
        options=["llama3", "phi3:mini"],
        index=0,
        help="llama3 gives more accurate answers; phi3:mini is faster but less reliable.",
    )
    st.caption(f"Active: `{selected_model}`")

st.divider()

with st.form(key="question_form"):
    question = st.text_area(
        "Your question",
        placeholder="e.g. Which region had the highest profit last year?",
        height=100,
    )
    submitted = st.form_submit_button("Ask", use_container_width=True)

if submitted:
    question = question.strip()
    if not question:
        st.warning("Please enter a question.")
    else:
        st.divider()
        st.subheader("Answer")
        try:
            placeholder = st.empty()
            full_response = ""
            with requests.post(
                "http://localhost:8000/ask/stream",
                json={"query": question, "model": selected_model},
                stream=True,
                timeout=120,
            ) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        placeholder.text(full_response)
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the backend. Make sure the FastAPI server is running on port 8000.")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

st.divider()
st.caption("Powered by ChromaDB · SentenceTransformers · Ollama (llama3 / phi3:mini)")
