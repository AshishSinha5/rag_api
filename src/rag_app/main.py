import argparse
from typing import Optional
from pathlib import Path

import torch

from contextlib import asynccontextmanager
from fastapi import FastAPI, Path as PathParam, Query, File, UploadFile
from pydantic import BaseModel, Field

from load_data import load_split_pdf_file, load_split_html_file, initialize_splitter
from load_llm import load_lamma_cpp
from vector_db import initialize_vector_db, create_vector_db, load_local_db
from utils import read_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fake_output(x: float):
    return "Answer to this query is 42"

ml_models = {}
db_name = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    model_path = "/home/models/llama.cpp/llama-2-7b.gguf.q8_0.bin"
    model_args = {'n_gpu_layers': 500,
                  'n_batch': 32,
                  'max_tokens': 100,
                  'n_ctx': 4096,
                  'temperature': 0.3,
                  'device': device}
    llm = load_lamma_cpp(model_path, model_args)
    ml_models["answer_to_everything"] = fake_output
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(
    title="RAG_APP",
    description="Retrival Augmented Generation APP which let's user upload a file and get the answer for the question using LLMs",
    lifespan=lifespan
)

@app.get("/")
def index():
    return {"message": "Hello World"}

text_splitter = initialize_splitter(chunk_size = 1000, chunk_overlap = 100)
model_name = "all-MiniLM-L6-v2"

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(f'../data/{file.filename}', 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    
    if file.filename.endswith('.pdf'):
        data = load_split_pdf_file(f'../data/{file.filename}', text_splitter)
    else:
        return {"message" : "Only PDF files permitted"}
    
    db = create_vector_db(data, model_name)


    return {"message": f"Successfully uploaded {file.filename}", 
            "num_splits" : len(data)}


@app.get("/query")
def query(query : str):
    persistant_dir = "../data/"
    collection_name = "test_collection"
    collection = load_local_db(persistant_dir, collection_name)
    results = collection.query(query_texts=[query], n_results = 2)
    return {"message": f"Query is {query}",
            "relavent_docs" : results}


if __name__ == "__main__":
    pass

