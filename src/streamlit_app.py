"""
Streamlit application for PDF based QA chatbot using RAG.
"""
import warnings
import logging
import os
import tempfile
import streamlit as st

# ==========================================
# 1. BROWSER & LOGGING CONFIGURATION
# ==========================================
# st.set_page_config MUST be the first Streamlit command.
st.set_page_config(page_title="PDF QA Chatbot", page_icon="📄🤖")

# Suppress noisy library logs
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

from loguru import logger
from dotenv import load_dotenv
load_dotenv()

# LangChain & Custom Imports
from document_loader import DocumentLoader
from text_splitter import split_text
from embeddings import HFEmbeddings
from vector_store import VectorStoreManager
from llm_client import LLMClient

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ==========================================
# 2. SIDEBAR & SETTINGS
# ==========================================
st.sidebar.title("🤖 LLM Settings")

# Update defaults to OpenAI
selected_provider = st.sidebar.selectbox("Provider", ["openai", "ollama"], index=0)
selected_model = st.sidebar.text_input("Model Name", value="gpt-4o" if selected_provider == "openai" else "llama3")

# Add API Key input
api_key = None
if selected_provider == "openai":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Get your key from platform.openai.com")

st.sidebar.subheader("Configuration")
st.sidebar.info(f"**Provider:** {selected_provider.capitalize()}")
st.sidebar.info(f"**Model:** {selected_model}")

# ==========================================
# 3. LLM INITIALIZATION
# ==========================================
@st.cache_resource # Cache the LLM so it doesn't reload on every click
def load_llm(provider, model, api_key=None):
    try:
        llm_wrapper = LLMClient(
            provider=provider,
            model_name=model,
            temperature=0.3,
            api_key=api_key
        )
        return llm_wrapper.get_llm()
    except Exception as e:
        st.error(f"Failed to connect to {provider}: {e}")
        return None

# Check if API key is provided before loading for OpenAI
llm = None
if selected_provider == "openai" and not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
else:
    llm = load_llm(selected_provider, selected_model, api_key)

# Stop the app execution if LLM isn't loaded yet
if llm is None:
    st.stop()
# ==========================================
# 4. CHAT HISTORY SETUP
# ==========================================
if 'store' not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# ==========================================
# 5. UI & FILE UPLOAD
# ==========================================
st.title("Document QA Assistant")
session_id = st.text_input("Session ID", value="default_session")
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
@st.cache_resource
def initialize_vector_db(_chunks):
    hf_wrapper = HFEmbeddings()
    vector_manager = VectorStoreManager(hf_wrapper)
    # This will now only run ONCE unless the chunks change
    return vector_manager.create_store(_chunks, "Chroma")
if uploaded_files:
    # --- Step 1: Load Documents ---
    all_docs = []
    for uploaded_file in uploaded_files:
        # Temporary save to process
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        docs_loader = DocumentLoader(tmp_path)
        all_docs.extend(docs_loader.PDF_loader())
        os.remove(tmp_path) # Cleanup

    # --- Step 2: Create Chunks & Vector Store ---
    chunks = split_text(all_docs)
    hf_wrapper = HFEmbeddings()
    vector_manager = VectorStoreManager(hf_wrapper)
    vectorDB = initialize_vector_db(chunks)
    retriever = vectorDB.as_retriever()

    # --- Step 3: Prompt Templates ---
    # System prompt to handle conversational context (History Re-phrasing)
    context_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context "
        "in the chat history, formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it."
    )
    
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", context_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # System prompt for actual QA
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use the retrieved context to answer. "
        "If you don't know the answer, say you don't know. Max 3 sentences.\n\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # --- Step 4: Building the RAG Chain ---
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # --- Step 5: Chat Execution ---
    user_input = st.chat_input("Ask a question about your documents:")
    if user_input:
        with st.spinner("Analyzing..."):
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            
            st.markdown("### Response:")
            st.write(response['answer'])
            
            with st.expander("View Chat History Details"):
                st.write(st.session_state.store[session_id].messages)