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

retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

# ===============================
# LLaMA-3 via Groq API
# ===============================
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # Groq LLaMA-3 model
    temperature=0.2,
    max_tokens=1024,
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

# ===============================
# MANUAL RAG FUNCTION
# ===============================
def ask_question(query: str) -> str:
    
    relevant_docs = retriever.invoke(query)

    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
You are a helpful assistant. Use the following context to answer the question.

CONTEXT:
{context_text}

QUESTION:
{query}

Answer only using the provided context.
If the answer is not present, say "Information not found in documents".
"""

    response = llm.invoke(prompt)

    return response.content

# ===============================
# TEST QUERY
# ===============================
print(ask_question("How much attendance should we have?"))





