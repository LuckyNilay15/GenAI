from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings=OpenAIEmbeddings(
    model="text-embedding-ada-002",
    dimension=32
)

result=embeddings.embed_query("Delhi is the capital of India")

print(str(result))



