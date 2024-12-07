from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request, Form
from typing import Annotated
import os
from dotenv import load_dotenv
import openai
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sentence_transformers import SentenceTransformer, util

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Load a pre-trained model for text similarity
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Cleaning: Remove special characters and unnecessary whitespace
    tokens = [word for word in tokens if word.isalnum()]

    # Normalization: Convert to lowercase
    tokens = [word.lower() for word in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return tokens

def perceptual_quality_analysis(prompt, response):
    # Measure the similarity between the prompt and the response
    prompt_embedding = similarity_model.encode(prompt, convert_to_tensor=True)
    response_embedding = similarity_model.encode(response, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(prompt_embedding, response_embedding).item()
    return similarity_score

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
                    if chunk.choices[0].delta.content is not None:
                        ai_response += chunk.choices[0].delta.content
                        await websocket.send_text(chunk.choices[0].delta.content)
                chat_responses.append(ai_response)

                # Perform perceptual quality analysis
                quality_score = perceptual_quality_analysis(user_input, ai_response)
                print(f"Perceptual Quality Score: {quality_score}")

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

        # Perform perceptual quality analysis
        quality_score = perceptual_quality_analysis(user_input, bot_response)
        print(f"Perceptual Quality Score: {quality_score}")

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