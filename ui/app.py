import streamlit as st
import requests

st.set_page_config(
    page_title="Superstore RAG Assistant",
    page_icon="🛒",
    layout="centered",
)

st.title("🛒 Superstore RAG Assistant")
st.markdown(
    "Ask any question about sales, customers, products, or regions "
    "and get an answer based on the Superstore dataset."
)
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
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://localhost:8000/ask",
                    json={"query": question},
                    timeout=120,
                )
                response.raise_for_status()
                answer = response.json().get("answer", "No answer returned.")
                st.divider()
                st.subheader("Answer")
                st.write(answer)
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the backend. Make sure the FastAPI server is running on port 8000.")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

st.divider()
st.caption("Powered by ChromaDB · SentenceTransformers · Ollama (llama3)")
