from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# model = ChatOllama(
#     model="gemma4:26b",
#     base_url = "http://172.31.95.149:11434",
#     # validate_model_on_init=True,
#     temperature=0.8,
#     # num_predict=256,
#     # other params ...
# )

model = ChatOpenAI(
    model="gemma4:26b",
    api_key="ollama",
    base_url="http://172.31.95.149:11434/v1",
    temperature=0.7,
    top_p=1
)

messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]
# model.invoke(messages)

print(model.invoke(messages))