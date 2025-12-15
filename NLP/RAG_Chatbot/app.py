"""
Streamlit UI for a Rock-Paper-Scissors RAG chatbot (QA model, lightweight).
"""

import streamlit as st
import sys
import os

# Add project root to import src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.retriever import load_retriever
from src.llm_pipeline import LlamaRAG

st.set_page_config(
    page_title="Rock-Paper-Scissors Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .main-header {text-align:center;padding:1rem;background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);color:white;border-radius:10px;margin-bottom:2rem;}
    .stChatMessage {background-color:#1e1e1e !important;border-radius:10px;padding:1rem;margin:0.5rem 0;}
    .stChatMessage p {color:#ffffff !important;}
    [data-testid="stChatMessageContent"] {background-color:#2d2d2d !important;color:#ffffff !important;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-header">
    <h1>🪨📄✂️ Rock-Paper-Scissors Assistant</h1>
    <p>Ask anything about the game.</p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    retriever, _ = load_retriever()
    llm = LlamaRAG(
        model_name="distilbert-base-uncased-distilled-squad",
        cache_dir="./models_cache",
        model_dir=None,
    )
    return retriever, llm


with st.spinner("Loading models..."):
    retriever, llm = load_models()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about rock-paper-scissors..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                contexts = retriever.invoke(prompt)
                response = llm.generate_response(question=prompt, contexts=contexts)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                msg = f"Error: {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        """
        - RAG over local data
        - QA model: distilbert-base-uncased-distilled-squad (260MB)
        - Vector store: Chroma
        - Embeddings: all-MiniLM-L6-v2
        """
    )
    st.markdown("### Suggested questions")
    for q in [
        "Quelles sont les règles du jeu ?",
        "Comment gagner plus souvent ?",
        "Quelle est l'histoire du jeu ?",
    ]:
        if st.button(q, key=q):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Local, lightweight RAG chatbot")
