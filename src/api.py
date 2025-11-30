from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag_chain import ask_with_history
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="../.env")

app = FastAPI(title="Gaiytri RAG API")

# CORS configuration - adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "https://gaiytri.com",  # Production domain
        "https://www.gaiytri.com",  # Production with www
        "http://gaiytri.com",  # HTTP version (if needed)
        "http://www.gaiytri.com",  # HTTP with www
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    chat_history: list = []


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
def health_check():
    return {
        "status": "healthy",
        "rag_initialized": True
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question to the RAG system with chat history (non-streaming)
    """
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Get answer from RAG chain with chat history
        answer = ask_with_history(
            question=request.question,
            chat_history=request.chat_history,
            stream=False
        )

        return AnswerResponse(
            answer=answer,
            success=True
        )

    except Exception as e:
        print(f"Error processing question: {e}")
        return AnswerResponse(
            answer="",
            success=False,
            error=f"An error occurred while processing your question: {str(e)}"
        )


@app.post("/ask/stream")
async def ask_question_stream(request: QuestionRequest):
    """
    Ask a question to the RAG system with chat history (streaming response)
    Uses Server-Sent Events for real-time streaming
    """
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def generate():
        try:
            # Get streaming generator from RAG chain
            stream_generator = ask_with_history(
                question=request.question,
                chat_history=request.chat_history,
                stream=True
            )

            # Stream each chunk as it arrives
            for chunk in stream_generator:
                # Extract content from the chunk
                if hasattr(chunk, 'content'):
                    content = chunk.content
                else:
                    content = str(chunk)

                # Send as SSE format
                if content:
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
