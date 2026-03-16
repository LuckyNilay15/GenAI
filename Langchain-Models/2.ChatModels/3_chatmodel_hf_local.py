from langchain_huggingface import ChatHuggingFace, HugggingFacePipeline

llm=HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,
    )
)

model=ChatHuggingFace(llm=llm)

result=model.invoke("What is the capital of Bihar?")

print(result.content)