from fastapi import FastAPI
from accelerate import Accelerator
from langchain_community.llms import CTransformers
from pydantic import BaseModel

app = FastAPI()

accelerator = Accelerator()

class Question(BaseModel):
    question: str

# model to download at root dir
# https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf

config = {'max_new_tokens': 2000, 'repetition_penalty': 1, 'context_length': 8000, 'temperature':0.3, 'gpu_layers':50}
# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = CTransformers(model="TheBloke/Llama-2-7b-Chat-GGUF", model_file="./llama-2-7b-chat.Q4_K_M.gguf", model_type="llama", gpu_layers=50, config=config)

llm, config = accelerator.prepare(llm, config)

@app.get("/")
def index():
    return {"message": "Welcome to the LLM generative API"}

@app.post("/completions")
async def chat(question: Question):
    answer = llm(question.question)
    return {"answer": answer}