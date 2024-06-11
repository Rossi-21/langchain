from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-2024-05-13",
    temperature=0.2,
    max_tokens=1000,
    verbose=True,
)
# response = llm.invoke("How are you?")
# print(response)

# response = llm.batch(
# ["What is your favorite type of sandwhich?", "Write a poem about AI."])
# print(response)

# response = llm.stream("Write a poem about AI.")

# for chunk in response:
#     print(chunk.content, end="", flush=True)
