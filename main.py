import os
import json
import asyncio
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import websockets

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return FileResponse("static/index.html")

def extract_reply_from_response(response_data: dict) -> str:
    """
    Extracts the reply text from a response.done event.
    Looks for output items from the assistant with transcript content.
    """
    outputs = response_data.get("output", [])
    for item in outputs:
        # Ensure this output is from the assistant
        if item.get("role") == "assistant" and "content" in item:
            # Look for a content part that has a transcript (audio modality)
            for part in item["content"]:
                if part.get("type") == "audio" and part.get("transcript"):
                    return part.get("transcript")
                # If there's also text content available, you can check for that as well.
                if part.get("type") == "text" and part.get("text"):
                    return part.get("text")
    return None

async def call_realtime_api(transcript: str) -> str:
    ws_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "OpenAI-Beta": "realtime=v1"
    }
    try:
        logging.debug(f"Connecting to {ws_url} with headers: {headers}")
        async with websockets.connect(ws_url, extra_headers=headers) as websocket:
            # Wait for the initial session event (e.g., session.created)
            initial_message = await asyncio.wait_for(websocket.recv(), timeout=10)
            data = json.loads(initial_message)
            logging.debug(f"Initial event received: {data}")

            # Now send our response.create event with the transcript
            event = {
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                    "instructions": transcript
                }
            }
            await websocket.send(json.dumps(event))
            logging.debug("Sent response.create event.")

            # Loop to wait for an event that contains the answer
            reply = None
            try:
                while True:
                    message = await asyncio.wait_for(websocket.recv(), timeout=20)
                    data = json.loads(message)
                    logging.debug(f"Received event: {data}")

                    event_type = data.get("type", "")
                    if event_type in ["response.updated", "response.complete", "response.done"]:
                        # For "response.done", the reply may be in the "response" object with an output list.
                        if event_type == "response.done" and "response" in data:
                            reply = extract_reply_from_response(data["response"])
                            if reply:
                                break
                        # For updated/complete events, check if there's a content field directly.
                        elif "response" in data and "content" in data["response"]:
                            reply = data["response"]["content"]
                            break
            except asyncio.TimeoutError:
                logging.error("Timeout waiting for response event.")
                reply = "Timeout waiting for a response from the realtime API."

            if reply is None:
                reply = "No valid reply received from the realtime API."
            return reply
    except Exception as e:
        logging.error("Error connecting to realtime API", exc_info=True)
        return f"Error connecting to realtime API: {str(e)}"

@app.post("/api/voice")
async def process_voice(request: Request):
    data = await request.json()
    transcript = data.get("transcript", "")
    reply = await call_realtime_api(transcript)
    return JSONResponse({"reply": reply})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

