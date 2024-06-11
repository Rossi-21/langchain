from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# Instaniate Model
model = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0.4)

prompt = ChatPromptTemplate.from_template("""
Answer the user's question:
Question: {input}
""")

chain = prompt | model
respone = chain.invoke({
    "input": "Hello"
})

print(respone)
