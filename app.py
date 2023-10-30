# First
import openai
import streamlit as st

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.chains import ConversationalRetrievalChain
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


chat_model = ChatOpenAI(openai_api_key=st.secrets["openai_apikey"], temperature = 0)

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_apikey"])

loader1 = UnstructuredHTMLLoader("mas1.html")
loader2 = UnstructuredHTMLLoader("mas4.html")
loader3 = UnstructuredHTMLLoader("mas6.html")
alldocs = []
document1 = loader1.load()
document2 = loader2.load()
document3 = loader3.load()
alldocs += document1
alldocs += document2
alldocs += document3

text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
texts = text_splitter.split_documents(alldocs)
db = Chroma.from_documents(texts, embeddings)

retriever = db.as_retriever()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(chat_model, retriever, memory=memory)

st.title("💬 Chatbot")

with st.sidebar:
    st.write(memory)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    st.chat_message("user").write(input)
    response = qa({"question": input})
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    st.chat_message("assistant").write(response["answer"])
    st.write(response)
