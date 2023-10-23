
# Using the retriaval qa prompt template from the langchain library - https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/retrieval_qa/prompt.py
def create_prompt(query, relevant_docs):
    relevant_text = ''
    relevant_docs = relevant_docs['documents'][0]
    for docs in relevant_docs:
        relevant_text += ("\n" + str(docs))

    prompt = f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{relevant_text}
Question: {query} 
Helpful Answer:"""
    
    return prompt