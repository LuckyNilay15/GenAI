from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

# Specify the model
llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

from pydantic import BaseModel, Field

# Define the schema
class Review(BaseModel):
    summary: str = Field(description="A brief summary of the software issue")

# Create the parser
parser = JsonOutputParser(pydantic_object=Review)

# Create the prompt with format instructions
prompt = ChatPromptTemplate.from_template(
    "Analyze the following software issue and provide a summary.\n{format_instructions}\nIssue: {request}"
)
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Chain model and parser
model = ChatHuggingFace(llm=llm)
chain = prompt | model | parser

# Invoke the chain
result = chain.invoke({"request": "There is a software bug hardware seems fluent but the software is not able to process the data properly"})

print(result)