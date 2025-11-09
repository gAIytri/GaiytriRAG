from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_chain import ask_with_history
import os
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
    Ask a question to the RAG system with chat history
    """
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Get answer from RAG chain with chat history
        answer = ask_with_history(
            question=request.question,
            chat_history=request.chat_history
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
