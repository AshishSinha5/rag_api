import argparse
from typing import Optional
from pathlib import Path
from pydantic import BaseModel
from pydantic import Field

import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, Path as PathParam, Query, File, UploadFile
from pydantic import BaseModel, Field

from load_data import load_split_pdf_file, load_split_html_file, initialize_splitter
from load_llm import load_lamma_cpp
from vector_db import create_vector_db, load_local_db
from prompts import create_prompt
from utils import read_file, load_yaml_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fake_output(x: float):
    return "Answer to this query is 42"

ml_models = {}
db_name = {}
model_args = load_yaml_file("llama2_config.yaml")
text_splitter = initialize_splitter(chunk_size = 1000, chunk_overlap = 100)
vector_db_model_name = "all-MiniLM-L6-v2"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    llm = load_lamma_cpp(model_args)
    ml_models["answer_to_query"] = llm
    # ml_models["answer_to_query"] = fake_output
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



# the model initialized when the app gets loaded but we can configure it if we want
@app.get("/init_llm")
def init_llama_llm(n_gpu_layers: int = Query(500, description="Number of layers to load in GPU"),
                n_batch: int = Query(32, description="Number of tokens to process in parallel. Should be a number between 1 and n_ctx."),
                max_tokens: int = Query(300, description="The maximum number of tokens to generate."),
                n_ctx: int = Query(4096, description="Token context window."),
                temperature: int = Query(0, description="Temperature for sampling. Higher values means more random samples.")):
    model_path = model_args["model_path"]
    model_args = {'model_path' : model_path,
                  'n_gpu_layers': n_gpu_layers,
                  'n_batch': n_batch,
                  'max_tokens': max_tokens,
                  'n_ctx': n_ctx,
                  'temperature': temperature,
                  'device': device}
    llm = load_lamma_cpp(model_args)
    ml_models["answer_to_query"] = llm
    return {"message": "LLM initialized"}

@app.post("/upload")
def upload_file(file: UploadFile = File(...), collection_name : Optional[str] = "test_collection"):
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
    elif file.filename.endswith('.html'):
        data = load_split_html_file(f'../data/{file.filename}', text_splitter)
    else:
        return {"message": "Only pdf and html files are supported"}
    
    db = create_vector_db(data, vector_db_model_name, collection_name)


    return {"message": f"Successfully uploaded {file.filename}", 
            "num_splits" : len(data)}


@app.get("/query")
def query(query : str, n_results : Optional[int] = 2, collection_name : Optional[str] = "test_collection"):
    try:
        collection_list = read_file('COLLECTIONS.txt')
        collection_list = collection_list.split("\n")[:-1]
    except Exception:
        return {"message": "No collections found uplaod some documents first"}

    if collection_name not in collection_list:
        return {"message": f"There is no collection with name {collection_name}",
                "available_collections" : collection_list}
    collection = load_local_db(collection_name)
    results = collection.query(query_texts=[query], n_results = n_results)
    prompt = create_prompt(query, results)
    output = ml_models["answer_to_query"](prompt)
    return {"message": f"Query is {query}",
            "relavent_docs" : results,
            "llm_output" : output}


if __name__ == "__main__":
    pass

