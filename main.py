from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from dotenv import load_dotenv
import requests
import os

load_dotenv()
openrouter_key = os.getenv("OPENROUTER_API_KEY")
hf_key = os.getenv("HUGGINGFACE_API_KEY")

# URLs
HF_MODEL_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
#MODEL_NAME = "openchat/openchat-3.5"
FREE_MODEL = "mistralai/mistral-7b-instruct"  # or openchat/openchat-3.5


# Headers
hf_headers = {"Authorization": f"Bearer {hf_key}"}
openrouter_headers = {
    "Authorization": f"Bearer {openrouter_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",
    "X-Title": "Relationship-AI"
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def is_message_unsafe(message: str) -> bool:
    keywords = ["unsafe", "hurt", "hit", "abuse", "toxic", "scared", "afraid", "violence"]
    return any(k in message.lower() for k in keywords)

class ChatRequest(BaseModel):
    message: str
    phase: str = "onboarding"

class AnalyzeRequest(BaseModel):
    message: str

class SummarizeRequest(BaseModel):
    history: List[Dict[str, str]]

@app.get("/")
def root():
    return {"msg": "✅ Backend running (OpenRouter + Hugging Face hybrid)"}

@app.post("/chat")
async def chat_with_ai(req: ChatRequest):
    try:
        prompt = f"""
You're a kind and emotionally intelligent relationship coach.

Phase: {req.phase}
User said: "{req.message}"

Respond with a thoughtful reflection or follow-up question.
"""

        payload = {
            "model": "mistralai/mistral-7b-instruct",  # ✅ FREE MODEL
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Relationship-AI"
        }

        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        res.raise_for_status()
        reply = res.json()["choices"][0]["message"]["content"]

        return {"response": reply.strip(), "alert": is_message_unsafe(req.message)}

    except Exception as e:
        print("❌ CHAT ERROR:", e)
        return {"error": str(e)}


@app.post("/analyze")
async def analyze_sentiment(data: AnalyzeRequest):
    try:
        prompt = f"Classify the sentiment of this message as POSITIVE, NEGATIVE, or NEUTRAL:\n{data.message}"
        
        payload = {
            "model": FREE_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        res = requests.post(OPENROUTER_URL, headers=openrouter_headers, json=payload)
        res.raise_for_status()
        sentiment = res.json()["choices"][0]["message"]["content"].strip().upper()

        if sentiment not in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
            sentiment = "UNKNOWN"

        depth = "deep" if len(data.message.split()) >= 10 else "shallow"
        return {"sentiment": sentiment, "depth": depth}

    except Exception as e:
        return {"error": f"Sentiment analysis failed: {str(e)}"}

        return {"error": f"Sentiment analysis failed: {str(e)}"}

@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    try:
        combined = "\n".join([f"User: {m['user']}\nAI: {m['ai']}" for m in req.history])
        prompt = f"""
Summarize this relationship coaching session in 3-4 lines. 
Highlight emotional tone, key reflections, and offer a gentle suggestion.

Session:
{combined}
"""

        payload = {
            "model": FREE_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }

        res = requests.post(OPENROUTER_URL, headers=openrouter_headers, json=payload)
        res.raise_for_status()
        summary = res.json()["choices"][0]["message"]["content"]

        return {"summary": summary.strip()}

    except Exception as e:
        return {"error": f"Summary failed: {str(e)}"}

