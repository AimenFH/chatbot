from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request, Form
from typing import Annotated
import os
from dotenv import load_dotenv
import openai

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_SECRET_KEY')

app = FastAPI()

templates = Jinja2Templates(directory="templates")

chat_responses = []

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "chat_responses": chat_responses})

chat_log = [{'role': 'system', 'content': 'you are Deep Learning and Virtual Reality Professor'}]

@app.websocket("/ws")
async def chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            user_input = await websocket.receive_text()
            chat_log.append({'role': 'user', 'content': user_input})
            chat_responses.append(user_input)

            try:
                response = openai.ChatCompletion.create(
                    model='gpt-4',
                    messages=chat_log,
                    temperature=0.6,
                    stream=True
                )

                ai_response = ''

                for chunk in response:
                    if chunk.choices[0].delta and chunk.choices[0].delta.content:
                        ai_response += chunk.choices[0].delta.content
                        await websocket.send_text(chunk.choices[0].delta.content)
                chat_responses.append(ai_response)

            except Exception as e:
                await websocket.send_text(f'Error: {str(e)}')
                break
    except WebSocketDisconnect:
        print("Client disconnected")

@app.post("/", response_class=HTMLResponse)
async def chat(request: Request, user_input: Annotated[str, Form()]):
    chat_log.append({'role': 'user', 'content': user_input})
    chat_responses.append(user_input)

    try:
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=chat_log,
            temperature=0.6
        )

        bot_response = response.choices[0].message.content
        chat_log.append({'role': 'assistant', 'content': bot_response})
        chat_responses.append(bot_response)

    except Exception as e:
        bot_response = f'Error: {str(e)}'
        chat_log.append({'role': 'assistant', 'content': bot_response})
        chat_responses.append(bot_response)

    return templates.TemplateResponse("home.html", {"request": request, "chat_responses": chat_responses})

@app.get("/image", response_class=HTMLResponse)
async def image_page(request: Request):
    print("Image page accessed")
    return templates.TemplateResponse("image.html", {"request": request})

@app.post("/image", response_class=HTMLResponse)
async def create_image(request: Request, user_input: Annotated[str, Form()]):
    try:
        response = openai.Image.create(
            prompt=user_input,
            n=1,
            size="256x256"
        )

        image_url = response.data[0].url
    except Exception as e:
        image_url = f'Error: {str(e)}'

    return templates.TemplateResponse("image.html", {"request": request, "image_url": image_url})