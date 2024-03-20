
from langchain_community.llms import CTransformers
from pydantic import BaseModel
from fastapi import FastAPI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

app = FastAPI()

class Question(BaseModel):
    question: str
# model to download at root dir
# https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf
def load_model() -> CTransformers:
    """ Load Llama Model"""
    config = {'max_new_tokens': 2000, 'repetition_penalty': 1, 'context_length': 8000, 'temperature':0.3, 'gpu_layers':50}
    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    Llama_model: CTransformers = CTransformers(
        model="./llama-2-7b-chat.Q4_K_M.gguf", 
        model_type="llama", 
        verbose= True,
        config=config
    )
    return Llama_model

llm = load_model()

template = """Question: {question}

Answer:"""

prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)

@app.get("/")
def index():
    return {"message": "Welcome to the LLM generative API"}

@app.post("/completions")
async def completions(question: Question):
    return llm_chain.invoke(question.question)
