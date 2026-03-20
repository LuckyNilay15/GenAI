# Now here we are going to make a embedding model for the document similarity.

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similairity
import numpy as np
load_dotenv()

embedding=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

documents=[
    "Virat Kohli is a great batsman known for aggressive batting and leadership.",
    "Rohit Sharma is a great batsman, hold a record of scoring double-century.",
    "Sachin Tendulkar is a great batsman, also known as God of cricket.",
    "Jasprit Bumrah is a great bowler known for orthodox actions and yorkers.",
    "MS Dhoni is the former indian captain known for his calm demaneur and finishing skills."
]

query='Tell me about virat kohli'

doc_embeddings=embedding.embed_documents(documents) #There will be 5 vectors. 
query_embedding=embedding.embed_query(query) # There will be 1 vector.

similarity=cosine_similarity([query_embedding],doc_embeddings)[0] # Pass both as 2D array.

index,score=sorted(list(enumerate(similarity),key=lambda x:x[1])[-1])

print(documents[index])
print("similarity score is",score)

# Now we are creating embeddings of documents which is static again and again rather store it in db known as vector databases.











