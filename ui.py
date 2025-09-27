import streamlit as st
import requests

st.set_page_config(page_title="🤖 Multi-Agent Demo", page_icon="🤖")

st.title("🤖 Multi-Agent AI Demo")
st.write("Ask a question and see how Retriever, Reasoning Agent, and Decision Agent work together.")

# User input
query = st.text_input("Enter your question:", placeholder="e.g. What is RAG?")

if st.button("Ask"):
    if query.strip():
        try:
            # Call your FastAPI backend
            response = requests.get("http://127.0.0.1:8000/ask", params={"query": query})
            
            if response.status_code == 200:
                result = response.json()
                st.subheader("🔍 Retrieved Docs")
                st.write(result["retrieval_agent_output"])
                
                st.subheader("🧠 Reasoning Agent")
                st.write(result["reasoning_agent_output"])
                
                st.subheader("✅ Decision Agent (Final Answer)")
                st.success(result["decision_agent_output"])
            else:
                st.error(f"Error: {response.status_code}")
        except Exception as e:
            st.error(f"Request failed: {e}")
    else:
        st.warning("Please enter a question.")
