from fastapi import FastAPI, Form
from typing import Annotated
from secret import openai_secret

openai = openai_secret
app = FastAPI()
chat_log = [{'role': 'system', 'content': 'You are Python Tutor.'}]

@app.post("/")
async def chat(user_input: Annotated[str, Form()]):

    chat_log.append({'role': 'user', 'content': user_input})
    response = openai.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=chat_log,
        temperature=0.6
    )


    bot_response = response.choices[0].message.content
    chat_log.append({'role': 'assistant', 'content': bot_response})
    return bot_response


