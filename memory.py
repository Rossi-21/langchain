from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.chat_message_histories.upstash_redis import (
    UpstashRedisChatMessageHistory,
)

from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-2024-05-13",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

URL = ""
TOKEN = ""
history = UpstashRedisChatMessageHistory(
    url=URL, token=TOKEN, ttl=500, session_id="chat1"
)

memory = ConversationBufferMemory(
    memory_key="chat_history"
)

# chain = prompt | model
chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    verbose=True,
)

msg = {
    "input": "Hello"
}

response = chain.invoke(msg)
print(response)
