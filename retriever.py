from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_retriever():
    # Small sample documents
    docs = [
        Document(page_content="Retrieval Augmented Generation (RAG) uses external knowledge to ground LLM outputs."),
        Document(page_content="An agent can use tools like retrievers, calculators, or APIs to complete tasks."),
        Document(page_content="Responsible AI includes fairness, bias mitigation, explainability, and safety guardrails.")
    ]

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Hugging Face embeddings (free)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever()
