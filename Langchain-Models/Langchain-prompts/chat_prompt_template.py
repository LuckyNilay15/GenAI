# Here we will be building a dynamic prompt template for a chatbot, i.e. we will be able to change the system message and the user message dynamically.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
# the above created an issue of where the dynamic variables are not being replaced by the values so we shifted towards the below method.

chat_template=ChatPromptTemplate.from_messages([
    ('system','You are a helpful {domain} expert'),
    ('human','Explain in simple terms what is {topic}')
])

prompt=chat_template.invoke({'domain':'cricket', 'topic':'Dusra'})

print(prompt)