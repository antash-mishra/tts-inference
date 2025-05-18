import asyncio
import base64
import json
import os
from typing import List
from dotenv import load_dotenv

import openai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ai import generate, translate

load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_audio_chunk(self, websocket: WebSocket, audio_data, sample_rate):
        # Convert audio data to base64 for sending over WebSocket
        audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
        await websocket.send_json({
            "audio": audio_base64,
            "sample_rate": sample_rate
        })

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get():
    with open("index.html", "r") as f:
        return f.read()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive text message from client
            data = await websocket.receive_text()
            print(f"Received message: {data}")
            
            # Parse JSON data
            request_data = json.loads(data)
            user_message = request_data.get("message", "")
            
            # Process with OpenAI
            response = client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=[
                    {"role": "system", "content": "You are a helpful voice assistant who responds in english only."},
                    {"role": "user", "content": user_message}
                ]
            )
            
            # Get response text from OpenAI
            ai_response = response.choices[0].message.content

            print(f"AI response: {ai_response}")
            
            # Translate the AI response to the target language
            src_lang, tgt_lang = "eng_Latn", "sat_Olck"
            ai_response = translate(ai_response, src_lang, tgt_lang)  # Uncomment if translation is needed
            
            # Send the AI response text back to the client
            await websocket.send_json({"text": ai_response})
            
            # Generate audio from the AI response
            description = "Sunita speaks slowly in a calm, moderate-pitched voice, delivering the news with a neutral tone. The recording is very high quality with no background noise."  # Voice description
            
            # Stream audio chunks back to client
            # for sample_rate, audio_chunk in generate(ai_response, description):
                # await manager.send_audio_chunk(websocket, audio_chunk, sample_rate)
            sample_rate, audio_chunk = generate(ai_response, description)
            await manager.send_audio_chunk(websocket, audio_chunk, sample_rate)
            print(f"Generated audio chunk of length: {len(audio_chunk)}")

            # Send end-of-stream marker
            await websocket.send_json({"eos": True})
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 