import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from parent directory
load_dotenv(dotenv_path="../.env")

DATA_PATH = "../data"
DB_PATH = "../db"

def run():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()

    print(f"📄 Loaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.metadata.get('source', 'unknown')}")

    # Better chunking strategy to preserve context
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Larger chunks to preserve context
        chunk_overlap=200,  # More overlap to maintain continuity
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    print(f"🔪 Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
    print("✅ Data ingested & embeddings stored successfully!")

if __name__ == "__main__":
    run()