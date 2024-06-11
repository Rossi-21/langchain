from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# Instaniate Model
llm = ChatOpenAI(
    model="gpt-4o-2024-05-13",
    temperature=0.2,
)

# === Prompt Template ===
# prompt = ChatPromptTemplate.from_template("Tell me a joke about a {subject}.")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate a list of 10 synonyms for the following word. Retun the results as a comma seperated list"),
        ("human", "{input}")
    ]
)
# Chreate LLM Chain
chain = prompt | llm


# response = chain.invoke({"subject": "dog"})
response = chain.invoke({"input": "happy"})
print(response)
