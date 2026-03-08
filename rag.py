import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# ===============================
# PDF FOLDER
# ===============================
PDF_FOLDER = "data"

documents = []

print("📂 Files found:", os.listdir(PDF_FOLDER))

# ===============================
# LOAD PDFs
# ===============================
for file in os.listdir(PDF_FOLDER):
    if file.lower().endswith(".pdf"):
        print(f"📄 Loading: {file}")
        loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
        docs = loader.load()
        documents.extend(docs)

print(f"✅ Loaded {len(documents)} pages")

# ===============================
# SPLIT TEXT
# ===============================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# ===============================
# EMBEDDINGS
# ===============================
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ===============================
# VECTOR STORE
# ===============================
vectorstore = FAISS.from_documents(chunks, embeddings)

# ===============================
# LLaMA-3 via Groq API
# ===============================
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # Groq LLaMA-3 model
    temperature=0.2,
    max_tokens=1024,
    groq_api_key=os.environ.get("GROQ_API_KEgsk_0dt50z4In2DBL2yqZfH9WGdyb3FYMuQjTCfQCR9QZ4AhTO9eiCSB")
)

# ===============================
# MANUAL RAG FUNCTION
# ===============================
def ask_question(query: str) -> str:
    # 1️⃣ Retrieve top 3 relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    try:
        relevant_docs = retriever.get_relevant_documents(query)
    except AttributeError:
        relevant_docs = retriever.invoke(query)
    
    if isinstance(relevant_docs, dict) and "docs" in relevant_docs:
        relevant_docs = relevant_docs["docs"]
    
    # 2️⃣ Build prompt
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"""
You are a helpful assistant. Use the following context to answer the question.

CONTEXT:
{context_text}

QUESTION:
{query}

Answer in a clear and concise way.
"""
    # 3️⃣ Generate answer using Groq LLaMA-3
    response = llm.invoke(prompt)  # just pass string directly
    return response

# ===============================
# TEST QUERY
# ===============================
print(ask_question("How much attendance should we have?"))




