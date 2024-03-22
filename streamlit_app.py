import streamlit as st
import asyncio
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.llms import CTransformers
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.base import AsyncCallbackHandler
from typing import AsyncIterable, Awaitable, Callable, Union, Any
from langchain.callbacks.manager import CallbackManager
from langchain.chains import LLMChain

Sender = Callable[[Union[str, bytes]], Awaitable[None]]

class AsyncStreamCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming, inheritance from AsyncCallbackHandler."""
    def __init__(self, send: Sender):
        super().__init__()
        self.send = send

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Rewrite on_llm_new_token to send token to client."""
        await self.send(f"data: {token}\n\n")


# app config
st.set_page_config(page_title="Chat bot", page_icon="🤖")
st.title("Chat bot")

template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """

prompt = ChatPromptTemplate.from_template(template)

callback = AsyncIteratorCallbackHandler()

def load_model():
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([callback])
    # Make sure the model path is correct for your system!
    llm: CTransformers = CTransformers(
            model="./llama-2-7b-chat.Q4_K_M.gguf", 
            model_type="llama", 
            verbose= True,
            callback_manager=callback_manager,
            config={'max_new_tokens': 2000, 'repetition_penalty': 1, 'context_length': 8000, 'temperature':0.3, 'gpu_layers':50, 'stream': True}
        )
    return llm

def get_response(user_query, chat_history): 
    model = load_model()
    parser = StrOutputParser()
    chain = prompt | model | parser
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot from Latitude. How can I help you?"),
    ]

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))