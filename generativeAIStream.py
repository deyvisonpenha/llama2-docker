import asyncio
from typing import AsyncIterable, Awaitable, Callable, Union, Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.base import AsyncCallbackHandler
from pydantic import BaseModel

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import CTransformers

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Sender = Callable[[Union[str, bytes]], Awaitable[None]]
 
template = """Let's work this out in a step by step way to be sure we have the right answer: {prompt}

    Answer: ."""

prompt = PromptTemplate(template=template, input_variables=["prompt"])

class AsyncStreamCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming, inheritance from AsyncCallbackHandler."""
    def __init__(self, send: Sender):
        super().__init__()
        self.send = send

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Rewrite on_llm_new_token to send token to client."""
        await self.send(f"data: {token}\n\n")


async def stream_message(message: str) -> AsyncIterable[str]:
    # Callbacks support token-wise streaming
    callback = AsyncIteratorCallbackHandler()
    callback_manager = CallbackManager([callback])
    # Make sure the model path is correct for your system!
    llm: CTransformers = CTransformers(
            model="./llama-2-7b-chat.Q4_K_M.gguf", 
            model_type="llama", 
            verbose= True,
            callback_manager=callback_manager,
            config={'max_new_tokens': 2000, 'repetition_penalty': 1, 'context_length': 8000, 'temperature':0.3, 'gpu_layers':50, 'stream': True}
        )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with an event to signal when it's done or an exception is raised."""
        try:
            await fn
        except Exception as e:
            # TODO: handle exception
            print(f"Caught exception: {e}")
        finally:
            # Signal the aiter to stop.
            event.set()

    # Begin a task that runs in the background.
    task = asyncio.create_task(wrap_done(
        llm_chain.arun(message),
        callback.done),
    )

    async for token in callback.aiter():
        # Use server-sent-events to stream the response
        yield token

    await task

def generate_message(message: str):
    # Make sure the model path is correct for your system!
    llm: CTransformers = CTransformers(
            model="./llama-2-7b-chat.Q4_K_M.gguf", 
            model_type="llama", 
            config={'max_new_tokens': 2000, 'repetition_penalty': 1, 'context_length': 8000, 'temperature':0.3, 'gpu_layers':50}
        )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.invoke(message)

class Request(BaseModel):
    """Request body for streaming."""
    prompt: str

@app.get("/")
def stream():
    return StreamingResponse(stream_message("tell me about yourself, a generative IA assistent."), media_type="text/event-stream")

@app.post("/stream_completions")
def stream(body: Request):
    return StreamingResponse(stream_message(body.prompt), media_type="text/event-stream")

@app.post("/completions")
def stream(body: Request):
    return generate_message(body.prompt)
