



def create_prompt(query, relevant_docs):
    relevant_text = ''
    relevant_docs = relevant_docs['documents'][0]
    for docs in relevant_docs:
        relevant_text += ("\n" + str(docs))

    prompt = f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{relevant_text}
Question: {query} 
Provide the steps in the following format and remove all mentions of figures and tables:
1 - ...
2 - ...
Steps:
Helpful Answer:"""
    
    return prompt