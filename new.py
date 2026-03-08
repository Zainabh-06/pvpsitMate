import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="pvpsitMate", page_icon="📚")

st.markdown("<h1>📚 pvpsitMate</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='caption'>AI-powered College Information Assistant</p>",
    unsafe_allow_html=True
)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("## 🎓 About pvpsitMate")
    st.write(
        """
        - RAG-based AI assistant  
        - Uses official PVPSIT PDFs  
        - Prevents hallucinations  
        - Built with LangChain + FAISS + Groq  
        """
    )

    st.markdown("## 🚀 Use Cases")
    st.write(
        """
        - Attendance criteria  
        - Program Outcomes (POs)  
        - Academic regulations  
        - College policies  
        """
    )

# -------------------------
# CSS Styling 
# -------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    background-repeat: no-repeat;
    background-position: center center;
    background-size: cover;
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}

h1 {
    text-align: center;
    font-weight: 700;
    color: #ffffff;
}

.caption {
    text-align: center;
    color: #cfd8dc;
    margin-bottom: 30px;
}

div[data-baseweb="input"] input {
    background-color: #1e293b;
    color: #ffffff;
    border-radius: 12px;
    padding: 12px;
    border: 1px solid #334155;
}

.answer-card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}

.user-msg {
    background: #2563eb;
    padding: 12px 16px;
    border-radius: 14px;
    margin-bottom: 10px;
    width: fit-content;
    max-width: 80%;
}

.ai-msg {
    background: #1e293b;
    padding: 14px 18px;
    border-radius: 14px;
    margin-top: 10px;
    max-width: 90%;
}

.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 14px;
    color: #cbd5e1;
}
</style>
""", unsafe_allow_html=True)


# -------------------------
# PDF Folder Path
# -------------------------
PDF_FOLDER = "data" # update if needed
# -------------------------
# Build / Load Vector Store
# -------------------------
@st.cache_resource
def build_vectorstore(pdf_folder):

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # If vectorstore already exists, load it
    if os.path.exists("vectorstore"):
        return FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Otherwise create it
    documents = []

    for file in os.listdir(pdf_folder):
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save FAISS index
    vectorstore.save_local("vectorstore")

    return vectorstore


with st.spinner("📥 Loading PDFs and building vector store..."):
    vectorstore = build_vectorstore(PDF_FOLDER)

st.success("✅ Answers are based on official college information")

# -------------------------
# Load Groq LLM
# -------------------------
@st.cache_resource
def get_llm():
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY environment variable not set!")
        return None

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=1024,
        groq_api_key=groq_api_key
    )


llm = get_llm()

# -------------------------
# RAG Question Function
# -------------------------
def ask_question(query: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    try:
        docs = retriever.get_relevant_documents(query)
    except AttributeError:
        docs = retriever.invoke(query)

    context_text = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are an academic assistant for PVPSIT.
Answer STRICTLY using the provided context.

CONTEXT:
{context_text}

QUESTION:
{query}

Answer clearly and directly.
"""

    response = llm.invoke(prompt)
    return response.content

# -------------------------
# User Input
# -------------------------
query = st.text_input(
    "How can I assist you today?",
    placeholder="e.g. What is PO6 of PVPSIT?"
)

if query and llm:
    with st.spinner("🤖 Searching official documents..."):
        try:
            answer = ask_question(query)

            st.markdown(
                f"<div class='user-msg'>🙋‍♂️ <b>You:</b> {query}</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<div class='answer-card ai-msg'>🤖 <b>pvpsitMate:</b><br>{answer}</div>",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error: {e}")



