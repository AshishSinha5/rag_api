import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import torch

import chromadb
from chromadb.utils import embedding_functions

# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chroma_client = chromadb.PersistentClient(path="../data")


# def initialize_vector_db(chroma_client, collection_name):
#     if len(chroma_client.list_collections()) > 0 and collection_name in [
#         chroma_client.list_collections()[0].name
#     ]:
#         chroma_client.delete_collection(name=collection_name)
#     else:
#         print(f"Creating collection: '{collection_name}'")
#         collection = chroma_client.create_collection(name=collection_name)
    
#     return collection

def register_collection(collection_name):
    # append to new line collection_name as string to COLLECTIONS.txt file
    with open("COLLECTIONS.txt", "a") as f:
        f.write(collection_name + "\n")


def create_vector_db(docs, model_name, collection_name):

    # create or load the vector store
    if len(chroma_client.list_collections()) > 0 and collection_name in [
        chroma_client.list_collections()[0].name
    ]:
        collection = chroma_client.get_collection(name=collection_name)
    else:
        # create the open-source embedding function
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name,
                                                                                     device = device)
        collection = chroma_client.create_collection(name=collection_name, embedding_function = embedding_function) 
        register_collection(collection_name)
         
    num_ids = collection.count()
    num_docs = len(docs)    
    collection.add(
        documents = [doc.page_content for doc in docs],
        ids = [f'id_{i}' for i in range(num_ids, num_ids + num_docs)],
        metadatas=  [doc.metadata for doc in docs]
    )

    return collection


def load_local_db(collection_name):
    # client = chromadb.PersistentClient(path=persist_directory)
    # collection = client.get_or_create_collection(name=collection_name)
    collection = chroma_client.get_collection(name=collection_name)
    return collection


