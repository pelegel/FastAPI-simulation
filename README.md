Hosting Gemma3 27b model using FastAPI.

To run the server:
uvicorn server:app --host 0.0.0.0 --port 5000

To call the server:
curl -N http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "מה קורה?", "session_id": "u1"}'

* Replace "session_id" with requested user ID to avoid shared conversation history between different users.
* Replace "prompt" with the requested user's input.

