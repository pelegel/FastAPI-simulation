from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import TorchAoConfig, Gemma3ForConditionalGeneration, AutoProcessor, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
import torch
import json
import threading
from typing import Dict
import uuid
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Free up CUDA memory
import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


app = FastAPI()


# ====== MODEL & TOKENIZER LOAD ======
model_id = "google/gemma-3-27b-it"

# Load the model in 4-bit quantization                               
quant_config = BitsAndBytesConfig(
    ## - *.weight of q_proj, k_proj, mlp -> torch.uint8 (4-bit quantized weights)
    ## - *.bias, *.norm.weight           -> torch.bfloat16 (higher precision values)   
    load_in_4bit=True,                      # Load model weights in 4-bit precision
    bnb_4bit_quant_type="nf4",              # Use Normal Float 4 (nf4) quantization type for better accuracy
    bnb_4bit_use_double_quant=True,         # Apply second-layer quantization to further compress weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computation

)

print("Loading model...")
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,                            
    torch_dtype=torch.bfloat16,              # Default dtype for unquantized parameters or operations
    device_map="auto",           
    quantization_config=quant_config,
    attn_implementation="flash_attention_2"  # Enable Flash Attention for speed
).eval()


processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
tokenizer = processor.tokenizer


with open("model_dtypes.txt", "w") as f:
    for name, param in model.named_parameters():
        f.write(f"{name}: {param.dtype}\n")



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
        yield json.dumps({"done": True}) + "\n"

    return StreamingResponse(stream_response(), media_type="application/json")


# ====== LAUNCH ======
# Run with: uvicorn server:app --host 0.0.0.0 --port 5000
