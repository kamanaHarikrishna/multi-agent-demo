🚀 Multi-Agent Demo (Streamlit + LangChain + FAISS)

This is a minimal demo of a multi-agent system built with Python and Streamlit.
The app simulates an agentic workflow where:

Retriever Agent fetches relevant chunks of information.

Reasoning Agent processes the retrieved context.

Decision Agent finalizes the output and serves it to the user.

It’s designed as a lightweight proof-of-concept for showcasing multi-agent orchestration in interviews and demos.


🎯 Example Usage

Input:
Explain what is RAG.
Output:
Retriever fetches context → reasoning agent refines → decision agent outputs a human-friendly explanation.

🛠️ Tech Stack

Python 3.10+
Streamlit – Web UI
LangChain – Agents & orchestration
FAISS – Vector database retriever
Free LLM (HuggingFace / Ollama) – No API key needed
