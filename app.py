import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import pipeline

# -----------------------------
# Setup: Retriever + LLM
# -----------------------------
def build_retriever():
    # Small sample knowledge base
    docs = [
        Document(page_content="Retrieval Augmented Generation (RAG) uses external knowledge to ground LLM outputs."),
        Document(page_content="An agent can use tools like retrievers, calculators, or APIs to complete tasks."),
        Document(page_content="Responsible AI includes fairness, bias mitigation, explainability, and safety guardrails.")
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever()

retriever = build_retriever()

# Free reasoning model
reasoning_model = pipeline("text2text-generation", model="google/flan-t5-small")

# -----------------------------
# Agent logic
# -----------------------------
def retrieval_tool(query, retriever):
    return retriever.get_relevant_documents(query)

def reasoning_agent(query, docs):
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer with reasoning:"
    result = reasoning_model(prompt, max_length=200, do_sample=False)
    return result[0]["generated_text"]

def decision_agent(answer, docs):
    if not docs:
        return "Not enough context to answer."
    return f"Final Answer: {answer}"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🤖 Multi-Agent Demo", page_icon="🤖")
st.title("🤖 Multi-Agent AI Demo")
st.write("Ask a question and see how retriever, reasoning agent, and decision agent work together.")

query = st.text_input("Enter your question:", placeholder="e.g. What is RAG?")

if st.button("Ask"):
    if query.strip():
        docs = retrieval_tool(query, retriever)
        answer = reasoning_agent(query, docs)
        final = decision_agent(answer, docs)

        st.subheader("🔍 Retrieved Docs")
        st.write([d.page_content for d in docs])

        st.subheader("🧠 Reasoning Agent")
        st.write(answer)

        st.subheader("✅ Decision Agent (Final Answer)")
        st.success(final)
    else:
        st.warning("Please enter a question.")
