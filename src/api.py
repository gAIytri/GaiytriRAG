from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
try:
    from rag_chain import ask_with_history
    from config import ENV_PATH
except ImportError:
    from src.rag_chain import ask_with_history
    from src.config import ENV_PATH
import os
import re
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Gaiytri RAG API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
cors_origins_str = os.getenv("CORS_ORIGINS", "")
cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()] if cors_origins_str else [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "https://gaiytri.com",
    "https://www.gaiytri.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


def strip_markdown(text):
    """Remove markdown formatting from text."""
    text = text.replace('**', '')
    text = text.replace('__', '')
    text = text.replace('##', '')
    text = text.replace('# ', '')
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    return text


class QuestionRequest(BaseModel):
    question: str
    chat_history: list = []

    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        v = v.strip()
        if len(v) == 0:
            raise ValueError('Question cannot be empty')
        if len(v) > 1000:
            raise ValueError('Question is too long. Please keep it under 1000 characters.')
        return v

    @field_validator('chat_history')
    @classmethod
    def validate_chat_history(cls, v):
        if len(v) > 20:
            # Keep only the last 20 messages to prevent context overflow
            return v[-20:]
        return v


class AnswerResponse(BaseModel):
    answer: str
    success: bool
    error: str = None


@app.get("/")
def read_root():
    return {
        "message": "Gaiytri RAG API",
        "status": "running",
        "endpoints": {
            "/ask": "POST - Ask a question about Gaiytri",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Check if the RAG system is operational."""
    health = {"status": "healthy", "components": {}}

    # Check ChromaDB
    try:
        try:
            from config import DB_PATH
        except ImportError:
            from src.config import DB_PATH
        db_exists = os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0
        health["components"]["vector_db"] = "ok" if db_exists else "missing"
    except Exception:
        health["components"]["vector_db"] = "error"
        health["status"] = "degraded"

    # Check Azure OpenAI connectivity
    try:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        if azure_endpoint and azure_key:
            health["components"]["llm"] = "configured"
        else:
            health["components"]["llm"] = "missing_credentials"
            health["status"] = "degraded"
    except Exception:
        health["components"]["llm"] = "error"
        health["status"] = "degraded"

    return health


@app.post("/ask", response_model=AnswerResponse)
@limiter.limit("10/minute")
async def ask_question(request: Request, body: QuestionRequest):
    """
    Ask a question to the RAG system with chat history (non-streaming)
    """
    try:
        # Get answer from RAG chain with chat history
        answer = ask_with_history(
            question=body.question,
            chat_history=body.chat_history,
            stream=False
        )

        return AnswerResponse(
            answer=answer,
            success=True
        )

    except Exception as e:
        print(f"Error processing question: {e}")
        return AnswerResponse(
            answer="I am having a bit of trouble right now. Please try again in a moment, or reach out to us directly at admin@gaiytri.com.",
            success=False,
            error=None
        )


@app.post("/ask/stream")
@limiter.limit("10/minute")
async def ask_question_stream(request: Request, body: QuestionRequest):
    """
    Ask a question to the RAG system with chat history (streaming response)
    Uses Server-Sent Events for real-time streaming
    """
    async def generate():
        try:
            # Get streaming generator from RAG chain
            stream_generator = ask_with_history(
                question=body.question,
                chat_history=body.chat_history,
                stream=True
            )

            # Stream each chunk as it arrives
            for chunk in stream_generator:
                # Extract content from the chunk
                if hasattr(chunk, 'content'):
                    content = chunk.content
                else:
                    content = str(chunk)

                # Send as SSE format with markdown stripped
                if content:
                    content = strip_markdown(content)
                    yield f"data: {json.dumps({'content': content})}\n\n"

            # Send completion signal
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            print(f"Error in streaming: {e}")
            error_msg = "I apologize, but I'm experiencing technical difficulties. Please try again or contact Gaiytri directly at admin@gaiytri.com"
            yield f"data: {json.dumps({'content': error_msg, 'error': True})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
