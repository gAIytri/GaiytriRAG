#!/bin/bash

echo "🚀 Starting Gaiytri RAG API Server..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please create a .env file with your OPENAI_API_KEY"
    exit 1
fi

# Check if database exists
if [ ! -d "db" ]; then
    echo "❌ Database not found!"
    echo "Please run 'python src/ingest.py' first to create the vector database"
    exit 1
fi

echo ""
echo "✅ Starting API server on http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd src
python api.py
