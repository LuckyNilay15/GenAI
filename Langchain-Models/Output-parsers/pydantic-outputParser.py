from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Load environment variables (ensure HUGGINGFACEHUB_ACCESS_TOKEN is set)
load_dotenv()

# Initialize the Large Language Model using HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

# Wrap the endpoint in ChatHuggingFace for easier chat-style interaction
model = ChatHuggingFace(llm=llm)

# 1. Define your desired data structure using Pydantic
# This allows for validation and ensures the LLM output follows this schema.
class Person(BaseModel):
    name: str = Field(description="The full name of the person")
    age: int = Field(description="The age of the person")
    city: str = Field(description="The city where the person lives")
    hobbies: List[str] = Field(description="A list of the person's hobbies")

# 2. Initialize the PydanticOutputParser with the Pydantic model
parser = PydanticOutputParser(pydantic_object=Person)

# 3. Create a PromptTemplate
# Use {format_instructions} to inject the parser's schema instructions into the prompt.
template = PromptTemplate(
    template="Generate a fictional person's profile.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. Create an LLM Chain (using LCEL - LangChain Expression Language)
# The chain flows from Template -> Model -> Parser
chain = template | model | parser

# 5. Execute the chain with an example query
query = "Create a profile for a young architect dynamic living in London."
result = chain.invoke({"query": query})

# Output the parsed result (it will be an instance of the Person Pydantic model)
print("Parsed Result Type:", type(result))
print("Name:", result.name)
print("Age:", result.age)
print("City:", result.city)
print("Hobbies:", result.hobbies)
print("\nFull Object:", result)
