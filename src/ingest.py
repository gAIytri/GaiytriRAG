import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
try:
    from config import DB_PATH, DATA_PATH, ENV_PATH
except ImportError:
    from src.config import DB_PATH, DATA_PATH, ENV_PATH

# Load environment variables from parent directory
load_dotenv(dotenv_path=ENV_PATH)

def run():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()

    print(f"Loaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.metadata.get('source', 'unknown')}")

    # Better chunking strategy to preserve context
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Larger chunks to preserve context
        chunk_overlap=200,  # More overlap to maintain continuity
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    print(f"Split into {len(chunks)} chunks")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    )
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
    print("Data ingested & embeddings stored successfully!")

if __name__ == "__main__":
    run()
