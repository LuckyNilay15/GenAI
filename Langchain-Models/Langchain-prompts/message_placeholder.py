from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

#chat template
chat_template=ChatPromptTemplate([
    ('system','You are a helpful assistant'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history=[]
#load chat history
with open('Langchain-prompts/chat_history.txt') as f:
    chat_history.extend(f.readlines())


#create prompt

prompt=chat_template.invoke({'chat_history':chat_history,'query':'What is the status of my refund?'})

print(prompt)




