from transformers import pipeline

# Load a free Hugging Face model (small + fast for demo)
reasoning_model = pipeline("text2text-generation", model="google/flan-t5-small")

def retrieval_tool(query, retriever):
    return retriever.get_relevant_documents(query)

# Agent 1: Reasoning
def reasoning_agent(query, docs):
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer with reasoning:"
    result = reasoning_model(prompt, max_length=200, do_sample=False)
    return result[0]["generated_text"]

# Agent 2: Decision
def decision_agent(answer, docs):
    if not docs:
        return "Not enough context to answer."
    return f"Final Answer: {answer}"
