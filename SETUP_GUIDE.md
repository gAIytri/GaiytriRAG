# Gaiytri RAG System - Setup Complete

## What You Have Built

A complete **Retrieval Augmented Generation (RAG)** system that allows AI to answer questions about Gaiytri LLC using your company documents.

## System Architecture

```
Company Documents (.md files)
    ↓
Ingestion Script (ingest.py)
    ↓
Vector Database (ChromaDB)
    ↓
RAG Chain (rag_chain.py) + Chat Memory
    ↓
FastAPI Backend (api.py)
    ↓
React Frontend (Vite)
```

## Files Overview

### 1. `ingest.py` - Data Ingestion
- Loads all `.md` files from `data/` folder
- Splits documents into 800-character chunks
- Creates embeddings using OpenAI API
- Stores in ChromaDB (`db/` folder)
- **Run once** when you add/update company documents

### 2. `rag_chain.py` - RAG Logic with Memory
- Loads the vector database
- Retrieves relevant document chunks (top 4)
- Sends to GPT-4o with context + chat history
- Professional executive-style responses
- Maintains conversation memory
- Returns AI-generated answer

### 3. `api.py` - FastAPI Backend
- RESTful API for web integration
- Handles chat history and context
- CORS configured for frontend
- Endpoints:
  - `GET /` - API information
  - `GET /health` - Health check
  - `POST /ask` - Ask questions with chat history

### 4. `app.py` - CLI Interface (Optional)
- Simple command-line chat interface
- Ask questions about Gaiytri
- Type 'quit' to exit

## How to Use

### First Time Setup (Already Done)
```bash
cd src
python ingest.py
# ✅ Data ingested & embeddings stored.
```

### Running the FastAPI Backend (For Website Integration)
```bash
cd src
python api.py
# Server runs on http://localhost:8000
```

API Features:
- **Chat Memory**: Remembers entire conversation until page refresh
- **Professional Tone**: Executive-style responses without markdown symbols
- **Context-Aware**: References previous messages naturally
- **CORS Enabled**: Works with React frontend

### Running the CLI (Optional - For Testing)
```bash
cd src
python app.py
```

Test questions:
- "What is Gaiytri?"
- "Who are the founders?"
- "What services does Gaiytri offer?"
- "What is the pricing?"

### When to Re-run Ingestion
Run `python ingest.py` again when you:
- Add new `.md` files to `data/`
- Update existing company documents
- Want to refresh the knowledge base

## Cost Estimates (OpenAI API)

### One-Time Costs
- **Ingestion**: ~$0.50 (for embedding 7 documents)
  - Only paid when you run `ingest.py`

### Runtime Costs (per query)
- **Embedding user question**: ~$0.0001
- **GPT-4o response**: ~$0.01-0.02
- **Total per query**: ~$0.01-0.02

### Monthly Estimates
- 100 queries: ~$1-2
- 500 queries: ~$5-10
- 1000 queries: ~$10-20

## Features

### ✅ Chat Memory
- Remembers entire conversation history
- LLM can reference previous questions and answers
- Natural follow-up questions work seamlessly
- Memory resets on page refresh

### ✅ Professional Responses
- Executive-style communication
- No markdown symbols or asterisks
- Concise 2-4 sentence answers
- Natural, conversational tone

### ✅ Context-Aware
- Combines company knowledge base with chat history
- References previous parts of conversation
- Understands "it", "that", "the first one" etc.

## Deployment Options

### Option 1: Current Setup (Recommended)
**FastAPI Backend** + **React Frontend**

Backend (`api.py` already created):
- Handles RAG queries with memory
- Professional AI responses
- CORS configured

Deploy to:
- **Render.com** (Free tier available)
- **Railway.app** (Free tier available)
- **AWS Lambda** (Pay per use)

### Option 2: Serverless
- Convert to AWS Lambda functions
- Deploy with Vercel Serverless Functions
- Pay only for actual usage

### Option 3: Docker Container
- Containerize both backend and frontend
- Deploy to any cloud provider
- Easier scaling and management

## Storage Requirements

### Local Storage
- Code: < 1MB
- Vector DB: ~500KB (current size)
- Dependencies: ~500MB (in venv)

### Deployed Storage
- **Minimal hosting**: Only need ~100MB
- **No model files** (uses OpenAI API)
- **Database grows**: +~50KB per additional document

## Security Notes

1. **Never commit `.env` file** to git
2. **Add to `.gitignore`**:
   ```
   .env
   db/
   db_local/
   venv/
   __pycache__/
   ```

3. **For production**, use environment variables:
   - Set `OPENAI_API_KEY` in hosting platform
   - Don't use `.env` file in production

## Next Steps

### Immediate
- ✅ System is ready to use
- ✅ Test with `python app.py`
- ✅ Try different questions

### Short-term
- Add more company documents to `data/`
- Create a simple web interface
- Deploy to a hosting platform

### Long-term
- Integrate with your website
- Add user authentication
- Track usage analytics
- Fine-tune prompts for better responses

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "OpenAI API Key Error"
Check `.env` file has:
```
OPENAI_API_KEY=sk-proj-your-key-here
```

### "Empty Response"
- Check if ingestion completed successfully
- Verify `db/` folder exists
- Re-run `python ingest.py`

## Support

- LangChain Docs: https://python.langchain.com/
- OpenAI API: https://platform.openai.com/docs
- ChromaDB: https://docs.trychroma.com/

---

**System Status**: ✅ Fully Operational with Web Integration
**Last Updated**: November 8, 2025
**Version**: 2.0 - FastAPI + Chat Memory + Professional Responses
