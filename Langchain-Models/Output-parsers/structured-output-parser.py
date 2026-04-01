from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

import os


load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model=ChatHuggingFace(llm=llm)

schema=[
    ResponseSchema(name="name",description="Name of the person"),
    ResponseSchema(name="age",description="Age of the person"),
    ResponseSchema(name="city",description="City of the person")
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template='Give the name, age and city of a fictional person \n {format_instructions}',
    input_variables=[],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

# prompt=template.invoke({})

# result=model.invoke(prompt)

# final_result=parser.parse(result.content)

chain = template | model | parser

final_result=chain.invoke({})

print(final_result)

