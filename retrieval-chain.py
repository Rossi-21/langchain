from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    print(len(splitDocs))
    return splitDocs


def create_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore


def create_chain(vectorStore):
    # Instaniate Model
    model = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0.4)

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}
    Question: {input}
    """)

    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever()
    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain


docs = get_documents_from_web(
    'https://python.langchain.com/docs/expression_language/')
vectorStore = create_db(docs)
chain = create_chain(vectorStore)


respone = chain.invoke({
    "input": "What is LCEL?"
})

print(respone["answer"])
