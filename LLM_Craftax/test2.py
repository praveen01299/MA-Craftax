import re
from ollama import Client

api_kwargs = {
    "messages": [{'role': 'user', 'content': 'generate a python function is_prime() to check if a number is a prime number. Give only the function and nothing else.'},],
    "model": 'gemma4:26b',
    # "max_completion_tokens": 1024,
    # "temperature":0.7,
}

# Connect to the remote server
client = Client(host='http://172.31.69.37:11434')
# client = Client(host='http://172.31.95.149:11434')

response = client.chat(**api_kwargs)
print(response.message.content)

def extract_code(text):
    # Pull code from markdown fences if present, otherwise use raw text
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

code = extract_code(response.message.content)

import ast
try:
    ast.parse(code)
except SyntaxError as e:
    print(f"Generated code has syntax error: {e}")
    exit(1)

with open("gen_func.py", "w") as f:
    f.write(code)

from gen_func import is_prime
print(is_prime(7))  # True
print(is_prime(10)) # False