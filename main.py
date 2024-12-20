from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from typing import Annotated
import os
from dotenv import load_dotenv
import openai
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sentence_transformers import SentenceTransformer, util
from lime.lime_text import LimeTextExplainer

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Load a pre-trained model for text similarity
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Cleaning: Remove special characters and unnecessary whitespace
    tokens = [word for word in tokens if word is not None and word.isalnum()]

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

# Initialize the LIME text explainer
explainer = LimeTextExplainer(class_names=['negative', 'positive'])

def explain_output(prompt, response):

    explanation = explainer.explain_instance(prompt, predict_fn, num_features=6)
    return explanation

def predict_fn(texts):
    # Function to predict the output for LIME
    responses = []
    for text in texts:
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[{'role': 'user', 'content': text}],
            temperature=0.6
        )
        responses.append(response.choices[0].message.content)
    return responses

def evaluate_image_quality(prompt, image_url):
    quality_score = 0.9
    return quality_score

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

        # Evaluate perceptual quality
        quality_score = evaluate_image_quality(user_input, image_url)
        print(f"Perceptual Quality Score: {quality_score}")

    except Exception as e:
        image_url = f'Error: {str(e)}'
        quality_score = None

    return templates.TemplateResponse("image.html", {"request": request, "image_url": image_url, "quality_score": quality_score})

@app.post("/feedback", response_class=HTMLResponse)
async def submit_feedback(request: Request, clarity: Annotated[str, Form()], creativity: Annotated[str, Form()], relevance: Annotated[str, Form()]):
    feedback = {
        "clarity": clarity,
        "creativity": creativity,
        "relevance": relevance
    }
    print(f"Feedback received: {feedback}")
    return templates.TemplateResponse("image.html", {"request": request, "feedback": feedback})

@app.post("/explain", response_class=HTMLResponse)
async def explain(request: Request, user_input: Annotated[str, Form()]):
    try:
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[{'role': 'user', 'content': user_input}],
            temperature=0.6
        )
        bot_response = response.choices[0].message.content

        # Generate LIME explanation
        explanation = explain_output(user_input, bot_response)
        explanation_html = explanation.as_html()

    except Exception as e:
        bot_response = f'Error: {str(e)}'
        explanation_html = ''

    return templates.TemplateResponse("explain.html", {"request": request, "user_input": user_input, "bot_response": bot_response, "explanation_html": explanation_html})