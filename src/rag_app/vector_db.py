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

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_vector_db(chroma_client, collection_name):
    if len(chroma_client.list_collections()) > 0 and collection_name in [
        chroma_client.list_collections()[0].name
    ]:
        chroma_client.delete_collection(name=collection_name)
    else:
        print(f"Creating collection: '{collection_name}'")
        collection = chroma_client.create_collection(name=collection_name)
    
    return collection


def create_vector_db(docs, model_name, collection_name):
    
    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name=model_name, 
                                                       model_kwargs= 
                                                       {'device' : device},
                                                       multi_process = True)
    # create db
    db = Chroma.from_documents(documents = docs, 
                               embedding = embedding_function, 
                               persist_directory = '../data',
                               collection_name = collection_name)
    return db


def load_local_db(persist_directory, collection_name):
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name=collection_name)
    return collection


