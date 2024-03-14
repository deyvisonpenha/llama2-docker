from fastapi import FastAPI
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel

app = FastAPI()

class Question(BaseModel):
    question: str

# Load the LlamaCpp language model, adjust GPU usage based on your hardware
llm = LlamaCpp(
    model_path="llama-2-7b-chat.Q4_0.gguf",
    n_gpu_layers=40,
    n_batch=512,  # Batch size for model processing
    verbose=False,  # Enable detailed logging for debugging
)

# Define the prompt template with a placeholder for the question
template = """
Question: {question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create an LLMChain to manage interactions with the prompt and model
llm_chain = LLMChain(prompt=prompt, llm=llm)

@app.get("/")
def index():
    return {"message": "Welcome to the LLM Chatbot API"}

@app.post("/api/chat")
async def chat(question: Question):
    answer = llm_chain.run(question.question)
    return {"answer": answer}