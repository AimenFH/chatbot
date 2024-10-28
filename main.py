from openai import OpenAI
from fastapi import FastAPI, Form
from typing import Annotated


openai = OpenAI(api_key='sk-54YzuGKKeUx4sgt50MpsA_FxPkGE0NJslg2ZS8GCF-T3BlbkFJ35Z35ntUZcDJSNTsDGH_x8fhIaVaTpCUr8Jcn5-AwA')
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


