import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
try:
    from config import DB_PATH, ENV_PATH
    from guardrails import is_off_topic, sanitize_input, validate_output, REDIRECT_RESPONSE
except ImportError:
    from src.config import DB_PATH, ENV_PATH
    from src.guardrails import is_off_topic, sanitize_input, validate_output, REDIRECT_RESPONSE

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH)

# Module-level cache
_cached_chain = None

def get_rag_chain(force_reload=False):
    global _cached_chain
    if _cached_chain is not None and not force_reload:
        return _cached_chain

    # Load vector database
    db = Chroma(persist_directory=DB_PATH, embedding_function=AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        request_timeout=15,
    ))
    # Increase k to retrieve more relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 6})

    # Initialize LLM with memory
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        temperature=0.7,
        request_timeout=30,
        max_retries=2,
    )

    # Create RAG prompt template with chat history and validation
    template = """You are Gaiytri AI, a helpful assistant for Gaiytri LLC, an AI automation company based in Jersey City, New Jersey. You are friendly, professional, and knowledgeable about Gaiytri.

Your role is to answer questions about Gaiytri LLC using the provided context. This includes questions about the company, its services, founders, technology, pricing, process, industries served, and contact information.

Guidelines:
- Answer based on the context provided below. If the context does not contain the answer, let the user know and suggest they contact admin@gaiytri.com for more details.
- For off-topic questions not related to Gaiytri, politely let them know you focus on Gaiytri-related topics and ask how you can help with Gaiytri.
- For greetings, respond warmly and briefly introduce yourself.
- Write in natural, conversational sentences only. Never use markdown formatting such as bold (**text**), italics (*text*), headers (#), bullet points, numbered lists, or any special formatting characters. Your response will be displayed as plain text.
- Keep responses concise, around 2 to 4 sentences for simple questions.

Context from our knowledge base:
{context}

Previous conversation:
{chat_history}

Current Question: {question}

Your response:"""

    prompt = ChatPromptTemplate.from_template(template)

    _cached_chain = (db, retriever, llm, prompt)
    return _cached_chain


def ask_with_history(question: str, chat_history: list = None, stream: bool = False):
    """
    Ask a question with chat history for conversational context
    Supports both streaming and non-streaming responses
    """
    # Pre-check: off-topic detection
    if is_off_topic(question):
        if stream:
            def redirect_gen():
                yield REDIRECT_RESPONSE
            return redirect_gen()
        return REDIRECT_RESPONSE

    # Sanitize input
    question = sanitize_input(question)

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

        # Get response from LLM - streaming or non-streaming
        if stream:
            # Return streaming generator
            return llm.stream(formatted_prompt)
        else:
            # Return complete response
            response = llm.invoke(formatted_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return validate_output(content)

    except Exception as e:
        print(f"Error in ask_with_history: {e}")
        # Fallback response without RAG context
        try:
            # Try to get LLM even if retriever failed
            llm = AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                temperature=0.7,
                request_timeout=30,
                max_retries=2,
            )
            if stream:
                return generate_fallback_response_stream(question, chat_history, llm)
            else:
                return generate_fallback_response(question, chat_history, llm)
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            fallback_msg = "I apologize, but I'm experiencing technical difficulties at the moment. Please contact Gaiytri directly for assistance. We appreciate your patience."
            if stream:
                # Return generator that yields the fallback message
                def fallback_generator():
                    yield fallback_msg
                return fallback_generator()
            else:
                return fallback_msg


def generate_fallback_response(question: str, chat_history: list, llm: AzureChatOpenAI):
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
        content = response.content if hasattr(response, 'content') else str(response)
        return validate_output(content)

    except Exception as e:
        print(f"Error in fallback response: {e}")
        return "I apologize, but I'm experiencing technical difficulties at the moment. Please contact Gaiytri directly at admin@gaiytri.com for assistance. We appreciate your patience."


def generate_fallback_response_stream(question: str, chat_history: list, llm: AzureChatOpenAI):
    """
    Generate a streaming response without RAG context when retrieval fails
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

        return llm.stream(formatted)

    except Exception as e:
        print(f"Error in fallback streaming response: {e}")
        def error_generator():
            yield "I apologize, but I'm experiencing technical difficulties at the moment. Please contact Gaiytri directly at admin@gaiytri.com for assistance. We appreciate your patience."
        return error_generator()
