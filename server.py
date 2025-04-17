from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import TorchAoConfig, Gemma3ForConditionalGeneration, AutoProcessor, AutoTokenizer, TextIteratorStreamer
import torch
import json
import threading
from typing import Dict
import uuid

app = FastAPI()

# ====== MODEL & TOKENIZER LOAD ======
model_id = "google/gemma-3-27b-it"
quant_config = TorchAoConfig("int4_weight_only", group_size=128)

print("Loading model...")
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quant_config
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id, padding_side="left")

# ====== MEMORY STORE FOR CHAT HISTORIES ======
conversations: Dict[str, list] = {}

DEFAULT_SYSTEM_PROMPT = {
    "role": "system",
    "content": [{"type": "text", "text": "תענה בבקשה על שאלות המשתמש בשפה העברית."}]
}

# ====== UTILS ======
def get_or_create_session(session_id: str) -> str:
    if session_id not in conversations:
        conversations[session_id] = [DEFAULT_SYSTEM_PROMPT]
    return session_id

def truncate_history(history, max_tokens=8192):
    # TODO: truncate based on tokenizer token count
    return history[-20:]  # simple fallback

# ====== ENDPOINT ======
@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    session_id = body.get("session_id", str(uuid.uuid4()))

    session_id = get_or_create_session(session_id)
    conversations[session_id].append({"role": "user", "content": [{"type": "text", "text": prompt}]})

    input_messages = truncate_history(conversations[session_id])

    inputs = processor.apply_chat_template(
        input_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def background_generate():
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                streamer=streamer
            )

    threading.Thread(target=background_generate).start()

    async def stream_response():
        output = ""
        for token in streamer:
            if token.strip():  # skip empty tokens
                output += token
                yield token
        # Save full reply to history
        conversations[session_id].append({
            "role": "assistant",
            "content": [{"type": "text", "text": output}]
        })
        yield json.dumps({"done": True, "full": output}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream_response(), media_type="application/json")

# ====== LAUNCH ======
# Run with: uvicorn server:app --host 0.0.0.0 --port 5000
