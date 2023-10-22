import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"


from langchain.llms import LlamaCpp

def load_lamma_cpp(model_args):
    llm = LlamaCpp(**model_args)

    return llm