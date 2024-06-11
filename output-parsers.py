from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()


# Instaniate Model
model = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0.7)


def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tell me a joke about the following subject"),
        ("human", "{input}")
    ])

    parser = StrOutputParser()

    chain = prompt | model | parser

    return chain.invoke({
        "input": "dog"
    })

# print(call_string_output_parser())


def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a list of 10 synonyms for the following word. Retun the results as a comma seperated list"),
        ("human", "{input}")
    ])

    parser = CommaSeparatedListOutputParser()

    # Chreate LLM Chain
    chain = prompt | model | parser

    return chain.invoke({
        "input": "happy"
    })


# print(call_list_output_parser())

def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Extract information from the following phrase. \nFormatting Instructions: {format_instructions}"),
        ("human", "{phrase}")
    ])

    class Person(BaseModel):
        name: str = Field(description="the name of the person")
        age: int = Field(description="the age of the person")

    parser = JsonOutputParser(pydantic_object=Person)

    chain = prompt | model | parser

    return chain.invoke({
        "phrase": "Max is 30 years old",
        "format_instructions": parser.get_format_instructions()
    })


print(call_json_output_parser())
