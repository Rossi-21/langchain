from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    # print(len(splitDocs))
    return splitDocs


def create_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore


def create_chain(vectorStore):
    # Instaniate Model
    model = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0.4)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    return retrieval_chain


def process_chat(chain, question, chat_history):
    respone = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    return respone["answer"]


if __name__ == '__main__':
    docs = get_documents_from_web(
        'https://python.langchain.com/docs/expression_language/')
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)

chat_history = [

]

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = process_chat(chain, user_input, chat_history)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    print("Assistant:", response)
