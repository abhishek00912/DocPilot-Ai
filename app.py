from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import InMemoryVectorStore


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocPilot AI",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ============================================================
# SESSION STATE
# ============================================================

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def document_process(path):

    # --------------------------------------------------------
    # 1. Load PDF
    # --------------------------------------------------------

    loader = PyPDFLoader(path)

    docs = loader.load()

    # --------------------------------------------------------
    # 2. Split PDF
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    docs = splitter.split_documents(docs)

    # --------------------------------------------------------
    # 3. HuggingFace Embeddings
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------
    # 4. Vector Store
    # --------------------------------------------------------

    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    # --------------------------------------------------------
    # 5. Save Vector Store
    # --------------------------------------------------------

    st.session_state.vector_db = vector_db

    st.session_state.document_uploaded = True

    return len(docs)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Document")

    st.write(
        "Upload a PDF and start chatting with your document."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Choose your PDF",
        type=["pdf"]
    )

    if uploaded_file:

        st.success(
            f"Selected: {uploaded_file.name}"
        )

        if st.button(
            "⚡ Process Document",
            use_container_width=True
        ):

            # Save PDF
            with open(
                "uploaded_document.pdf",
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getvalue()
                )

            # Reset previous chat
            st.session_state.chat_history = []

            # Reset previous vector store
            st.session_state.vector_db = None

            st.session_state.document_uploaded = False

            # Save file name
            st.session_state.file_name = (
                uploaded_file.name
            )

            # Process document
            with st.spinner(
                "📖 Reading and understanding your PDF..."
            ):

                chunk_count = document_process(
                    "./uploaded_document.pdf"
                )

            st.success(
                "✅ PDF processed successfully!"
            )

            st.caption(
                f"📚 {chunk_count} text chunks created"
            )

    st.divider()

    # ========================================================
    # RAG PIPELINE
    # ========================================================

    st.subheader("🔎 RAG Pipeline")

    st.write("📄 PDF")
    st.write("↓")
    st.write("📖 PyPDFLoader")
    st.write("↓")
    st.write("✂️ Recursive Text Splitter")
    st.write("↓")
    st.write("🤗 HuggingFace Embeddings")
    st.write("↓")
    st.write("🗂️ InMemoryVectorStore")
    st.write("↓")
    st.write("🔎 Similarity Search")
    st.write("↓")
    st.write("🦙 Ollama Llama 3.2")

    st.divider()

    st.caption(
        "DocPilot AI • HuggingFace + Ollama"
    )


# ============================================================
# MAIN PAGE
# ============================================================

st.title("📚 DocPilot AI")

st.caption(
    "Your intelligent PDF assistant — "
    "ask questions directly from your document."
)

st.divider()


# ============================================================
# HOME SCREEN
# ============================================================

if not st.session_state.document_uploaded:

    st.info(
        "👈 Upload a PDF from the sidebar to get started."
    )

    st.subheader(
        "🚀 Your PDF, now interactive"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📄 Input",
            "PDF"
        )

    with col2:

        st.metric(
            "🤗 Embeddings",
            "HuggingFace"
        )

    with col3:

        st.metric(
            "🦙 LLM",
            "Ollama"
        )

    st.divider()

    st.subheader(
        "💡 What can you ask?"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            """
            📖 **Understand your document**

            - What is this document about?
            - What are the main points?
            - Summarize this document.
            """
        )

    with col2:

        st.write(
            """
            🔎 **Find information**

            - Explain a specific topic.
            - Find important information.
            - Ask questions about the PDF.
            """
        )


# ============================================================
# CHAT INTERFACE
# ============================================================

if (
    st.session_state.document_uploaded
    and st.session_state.vector_db is not None
):

    st.success(
        f"📗 **{st.session_state.file_name}** "
        "is ready for questions."
    )

    st.subheader(
        "💬 Chat with your PDF"
    )
    if st.button("🗑️ Clear Chat", use_container_width=True):
     st.session_state.chat_history = []
     st.rerun()

    # --------------------------------------------------------
    # Display Chat History
    # --------------------------------------------------------

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

    # --------------------------------------------------------
    # Chat Input
    # --------------------------------------------------------

    query = st.chat_input(
        "Ask anything about your PDF..."
    )

    if query:

        # ====================================================
        # USER MESSAGE
        # ====================================================

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):

            st.write(query)

        # ====================================================
        # SIMILARITY SEARCH
        # ====================================================

        with st.spinner(
            "🔎 Searching your document..."
        ):

            documents = (
                st.session_state.vector_db
                .similarity_search(
                    query,
                    k=6
                )
            )

        # ====================================================
        # BUILD CONTEXT
        # ====================================================

        context = ""

        for doc in documents:

            context += (
                doc.page_content
                + "\n\n"
            )

        # ====================================================
        # PROMPT
        # ====================================================

        prompt = f"""
You are DocPilot AI, an intelligent PDF assistant.

Answer the user's question using ONLY the
information provided in the context.

Do NOT use outside knowledge.

If the answer is not available in the context,
respond exactly:

"I couldn't find the answer in the uploaded PDF."

Keep the answer clear, accurate and easy to understand.

Context:
------------------------------
{context}
------------------------------

Question:
{query}
"""

        # ====================================================
        # OLLAMA RESPONSE
        # ====================================================

        with st.chat_message("assistant"):

            with st.spinner(
                "🤖 DocPilot is thinking..."
            ):

                result = llm.invoke(
                    prompt
                )

            # Ollama normally returns the answer
            # directly as a string.

            answer = result.content

            st.write(
                answer
            )

        # ====================================================
        # SAVE ASSISTANT RESPONSE
        # ====================================================

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )