from langchain_core.prompts import PromptTemplate

template=PromptTemplate(
    input_variables=['paper_input','paper_style','length_input'],
    template="""
    You are a helpful assistant.
    Summarize the following {paper_input} in {paper_style} style and in {length_input} words.
    """
)

template.save('template.json')

