from dotenv import load_dotenv
import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import BM25Retriever


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GET GROQ API KEY
# Works locally with .env
# Works on Streamlit Cloud with Secrets
# ============================================================

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocPilot AI",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CHECK API KEY
# ============================================================

if not groq_api_key:

    st.error(
        "❌ GROQ_API_KEY not found."
    )

    st.info(
        """
        For local development:

        Create a `.env` file and add:

        GROQ_API_KEY=your_new_groq_key

        For Streamlit Cloud:

        Add GROQ_API_KEY inside App Settings → Secrets.
        """
    )

    st.stop()


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=groq_api_key
)


# ============================================================
# SESSION STATE
# ============================================================

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "vector_retriever" not in st.session_state:
    st.session_state.vector_retriever = None

if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    vector_retriever,
    bm25_retriever,
    k=8
):

    # --------------------------------------------------------
    # Vector Search
    # --------------------------------------------------------

    vector_docs = vector_retriever.invoke(
        query
    )

    # --------------------------------------------------------
    # BM25 Search
    # --------------------------------------------------------

    bm25_docs = bm25_retriever.invoke(
        query
    )

    # --------------------------------------------------------
    # Reciprocal Rank Fusion
    # --------------------------------------------------------

    scores = {}
    documents = {}

    # Vector search weight = 0.65

    for rank, doc in enumerate(vector_docs):

        content = doc.page_content

        if content not in scores:
            scores[content] = 0

        if content not in documents:
            documents[content] = doc

        scores[content] += (
            0.65 / (rank + 1)
        )

    # BM25 search weight = 0.35

    for rank, doc in enumerate(bm25_docs):

        content = doc.page_content

        if content not in scores:
            scores[content] = 0

        if content not in documents:
            documents[content] = doc

        scores[content] += (
            0.35 / (rank + 1)
        )

    # --------------------------------------------------------
    # Rank Documents
    # --------------------------------------------------------

    ranked_documents = sorted(
        documents.values(),
        key=lambda doc: scores[
            doc.page_content
        ],
        reverse=True
    )

    return ranked_documents[:k]


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
        chunk_size=800,
        chunk_overlap=120
    )

    docs = splitter.split_documents(
        docs
    )

    # --------------------------------------------------------
    # 3. HuggingFace Embeddings
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    )

    # --------------------------------------------------------
    # 4. Vector Store
    # --------------------------------------------------------

    vector_db = (
        InMemoryVectorStore.from_documents(
            documents=docs,
            embedding=embeddings
        )
    )

    # --------------------------------------------------------
    # 5. Vector Retriever
    # --------------------------------------------------------

    vector_retriever = (
        vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 8
            }
        )
    )

    # --------------------------------------------------------
    # 6. BM25 Retriever
    # --------------------------------------------------------

    bm25_retriever = (
        BM25Retriever.from_documents(
            docs,
            k=8
        )
    )

    # --------------------------------------------------------
    # 7. Save Components
    # --------------------------------------------------------

    st.session_state.vector_db = vector_db

    st.session_state.vector_retriever = (
        vector_retriever
    )

    st.session_state.bm25_retriever = (
        bm25_retriever
    )

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

            # ------------------------------------------------
            # Save PDF
            # ------------------------------------------------

            with open(
                "uploaded_document.pdf",
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getvalue()
                )

            # ------------------------------------------------
            # Reset Chat
            # ------------------------------------------------

            st.session_state.chat_history = []

            # ------------------------------------------------
            # Reset Retrieval
            # ------------------------------------------------

            st.session_state.vector_db = None

            st.session_state.vector_retriever = None

            st.session_state.bm25_retriever = None

            st.session_state.document_uploaded = False

            # ------------------------------------------------
            # Save File Name
            # ------------------------------------------------

            st.session_state.file_name = (
                uploaded_file.name
            )

            # ------------------------------------------------
            # Process PDF
            # ------------------------------------------------

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
    st.write("🔎 Hybrid Search")
    st.write("↓")
    st.write("⚡ Groq LLM")

    st.divider()

    st.caption(
        "DocPilot AI • HuggingFace + Groq"
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
            "⚡ LLM",
            "Groq"
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

    # --------------------------------------------------------
    # PDF READY
    # --------------------------------------------------------

    st.success(
        f"📗 **{st.session_state.file_name}** "
        "is ready for questions."
    )

    st.subheader(
        "💬 Chat with your PDF"
    )

    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()

    # --------------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

    # --------------------------------------------------------
    # USER INPUT
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
        # HYBRID SEARCH
        # ====================================================

        with st.spinner(
            "🔎 Searching your document..."
        ):

            documents = hybrid_search(
                query=query,
                vector_retriever=(
                    st.session_state
                    .vector_retriever
                ),
                bm25_retriever=(
                    st.session_state
                    .bm25_retriever
                ),
                k=8
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

The context was retrieved using:

1. Semantic vector search
2. BM25 keyword search
3. Hybrid ranking

Use the retrieved context carefully.

Do NOT use outside knowledge.

Do NOT guess or make up information.

If the answer is available in the provided
context, answer the question directly.

If the answer is not available in the context,
respond exactly:

"I couldn't find the answer in the uploaded PDF."

Keep the answer clear, accurate and concise.

Context:
------------------------------
{context}
------------------------------

Question:
{query}
"""

        # ====================================================
        # GROQ RESPONSE
        # ====================================================

        with st.chat_message("assistant"):

            with st.spinner(
                "🤖 DocPilot is thinking..."
            ):

                result = llm.invoke(
                    prompt
                )

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