RAG_SYSTEM_PROMPT = (
    "You are a helpful sales data assistant. "
    "Answer questions using only the context provided. "
    "If the answer is not in the context, say you don't know."
)

RAG_USER_PROMPT = (
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer using only the context."
)

