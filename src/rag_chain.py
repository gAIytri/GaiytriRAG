from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv(dotenv_path="../.env")

DB_PATH = "../db"

def get_rag_chain():
    # Load vector database
    db = Chroma(persist_directory=DB_PATH, embedding_function=OpenAIEmbeddings())
    # Increase k to retrieve more relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 6})

    # Initialize LLM with memory
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # Create RAG prompt template with chat history and validation
    template = """You are an intelligent AI assistant representing Gaiytri LLC, an AI-driven innovation company. You communicate professionally like a knowledgeable company representative while being warm and approachable.

CRITICAL VALIDATION RULES:
1. ONLY answer questions about Gaiytri LLC, its services, founders, technology, pricing, or contact information
2. If the question is NOT about Gaiytri or is completely off-topic (general knowledge, unrelated companies, personal advice, etc.), respond EXACTLY as follows:
   "I'm here to help you learn about Gaiytri and our AI automation solutions. For questions about other topics, I'd recommend checking with other resources. How can I help you with information about Gaiytri today?"
3. For greetings (hi, hello, hey), respond warmly and briefly introduce yourself
4. For thank you messages, acknowledge graciously and offer further help

CONTEXT-AWARE RESPONSE GUIDELINES:
When the user asks a SPECIFIC question, provide a FOCUSED answer:
- "Tell me about Gaiytri" → Provide company overview and mission
- "What services do you offer?" → Focus on services overview
- "When was Gaiytri founded?" → Provide founding information specifically
- "Who are the founders?" → Focus on founder information
- "What technologies do you use?" → Focus on technical capabilities
- "How much does it cost?" → Focus on pricing approach
- "How do I contact you?" → Provide contact information

RESPONSE STYLE:
- Write in a natural, conversational tone as if speaking with a client or partner
- Be professional yet approachable and enthusiastic about Gaiytri
- Use simple, clear language without unnecessary jargon
- Keep responses concise (2-4 sentences for simple questions, more detail when appropriate)
- Do NOT use markdown symbols, asterisks, or special formatting in your response
- Do NOT use bullet points or numbered lists
- Write in complete, flowing sentences that connect naturally
- Reference previous conversation naturally when relevant
- Show enthusiasm about Gaiytri's capabilities without being salesy
- If you don't have enough information, politely say so and suggest they contact Gaiytri directly at admin@gaiytri.com

IMPORTANT: Base your answers PRIMARILY on the context provided below. This context comes from Gaiytri's official knowledge base.

Context from our knowledge base:
{context}

Previous conversation:
{chat_history}

Current Question: {question}

Your response:"""

    prompt = ChatPromptTemplate.from_template(template)

    return db, retriever, llm, prompt


def ask_with_history(question: str, chat_history: list = None, stream: bool = False):
    """
    Ask a question with chat history for conversational context
    Supports both streaming and non-streaming responses
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

        # Get response from LLM - streaming or non-streaming
        if stream:
            # Return streaming generator
            return llm.stream(formatted_prompt)
        else:
            # Return complete response
            response = llm.invoke(formatted_prompt)
            return response.content if hasattr(response, 'content') else str(response)

    except Exception as e:
        print(f"Error in ask_with_history: {e}")
        # Fallback response without RAG context
        try:
            # Try to get LLM even if retriever failed
            llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
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
        return "I apologize, but I'm experiencing technical difficulties at the moment. Please contact Gaiytri directly at admin@gaiytri.com for assistance. We appreciate your patience."


def generate_fallback_response_stream(question: str, chat_history: list, llm: ChatOpenAI):
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