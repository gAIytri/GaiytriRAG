from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv(dotenv_path="../.env")

DB_PATH = "../db"

def get_rag_chain():
    # Load vector database
    db = Chroma(persist_directory=DB_PATH, embedding_function=OpenAIEmbeddings())
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # Initialize LLM with memory
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # Create RAG prompt template with chat history
    template = """You are a professional representative of Gaiytri LLC, communicating like a company executive. Your role is to provide clear, concise, and professional information about Gaiytri.

Guidelines for your responses:
- Write in a natural, conversational tone as if speaking with a client or partner
- Be professional yet approachable
- Use simple, clear language without unnecessary jargon
- Keep responses concise and to the point (2-4 sentences unless more detail is needed)
- Do NOT use markdown symbols, asterisks, or special formatting
- Do NOT use bullet points or numbered lists unless absolutely necessary
- Write in complete, flowing sentences
- Reference previous parts of the conversation naturally when relevant
- If you don't have enough information, politely say so and suggest they contact Gaiytri directly

Context from our knowledge base:
{context}

Chat History:
{chat_history}

Current Question: {question}

Response:"""

    prompt = ChatPromptTemplate.from_template(template)

    return db, retriever, llm, prompt


def ask_with_history(question: str, chat_history: list = None):
    """
    Ask a question with chat history for conversational context
    """
    try:
        db, retriever, llm, prompt = get_rag_chain()

        # Format documents from retriever
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Format chat history
        def format_chat_history(history):
            if not history:
                return "No previous conversation."

            formatted = []
            for msg in history:
                role = "User" if msg.get("role") == "user" else "Assistant"
                formatted.append(f"{role}: {msg.get('content', '')}")
            return "\n".join(formatted)

        # Get relevant documents - use invoke() for newer LangChain versions
        try:
            docs = retriever.invoke(question)
        except AttributeError:
            # Fallback for older versions
            docs = retriever.get_relevant_documents(question)

        context = format_docs(docs) if docs else "No relevant information found in our knowledge base."

        # Build the prompt
        formatted_prompt = prompt.format(
            context=context,
            chat_history=format_chat_history(chat_history or []),
            question=question
        )

        # Get response from LLM
        response = llm.invoke(formatted_prompt)

        return response.content if hasattr(response, 'content') else str(response)

    except Exception as e:
        print(f"Error in ask_with_history: {e}")
        # Fallback response without RAG context
        try:
            # Try to get LLM even if retriever failed
            llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
            return generate_fallback_response(question, chat_history, llm)
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return "I apologize, but I'm experiencing technical difficulties at the moment. Please contact Gaiytri directly for assistance. We appreciate your patience."


def generate_fallback_response(question: str, chat_history: list, llm: ChatOpenAI):
    """
    Generate a response without RAG context when retrieval fails
    """
    try:
        fallback_template = """You are a professional representative of Gaiytri LLC.

I apologize, but I'm having trouble accessing our knowledge base at the moment. However, I can still help you.

Chat History:
{chat_history}

Question: {question}

Please provide a helpful response based on what you know about typical business services and suggest they contact Gaiytri directly for specific details.

Response:"""

        def format_chat_history(history):
            if not history:
                return "No previous conversation."
            formatted = []
            for msg in history:
                role = "User" if msg.get("role") == "user" else "Assistant"
                formatted.append(f"{role}: {msg.get('content', '')}")
            return "\n".join(formatted)

        fallback_prompt = ChatPromptTemplate.from_template(fallback_template)
        formatted = fallback_prompt.format(
            chat_history=format_chat_history(chat_history or []),
            question=question
        )

        response = llm.invoke(formatted)
        return response.content if hasattr(response, 'content') else str(response)

    except Exception as e:
        print(f"Error in fallback response: {e}")
        return "I apologize, but I'm experiencing technical difficulties at the moment. Please contact Gaiytri directly at info@gaiytri.com for assistance. We appreciate your patience."