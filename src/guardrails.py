import re

# Topics that should be redirected
OFF_TOPIC_PATTERNS = [
    r'\b(write|generate|create)\s+(code|script|program|essay|story|poem)\b',
    r'\b(hack|exploit|attack|phishing|malware|virus)\b',
    r'\b(political|politics|election|democrat|republican|trump|biden)\b',
    r'\b(medical\s+advice|diagnos|prescri|symptom)\b',
    r'\b(legal\s+advice|lawsuit|attorney|lawyer)\b',
    r'\b(stock\s+tip|investment\s+advice|crypto|bitcoin|ethereum)\b',
    r'\b(personal\s+opinion|believe|think\s+about)\b(?!.*gaiytri)',
]

REDIRECT_RESPONSE = (
    "I appreciate your curiosity! I am here specifically to help you learn about "
    "Gaiytri and our AI automation solutions. For questions about other topics, "
    "I would recommend checking with other resources. "
    "How can I help you with information about Gaiytri today?"
)

def is_off_topic(question: str) -> bool:
    """Check if the question is clearly off-topic for a Gaiytri assistant."""
    question_lower = question.lower()

    # Allow greetings
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'thanks', 'thank you']
    if question_lower.strip() in greetings or len(question_lower.strip()) < 4:
        return False

    # Check for off-topic patterns
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, question_lower):
            return True

    return False


def sanitize_input(text: str) -> str:
    """Clean user input to prevent injection attempts."""
    # Remove potential prompt injection markers
    injection_patterns = [
        r'ignore\s+(previous|above|all)\s+(instructions|prompts)',
        r'you\s+are\s+now\s+',
        r'system\s*:\s*',
        r'<\|.*?\|>',
        r'\[INST\]',
        r'\[\/INST\]',
    ]
    cleaned = text
    for pattern in injection_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


def validate_output(response: str) -> str:
    """Post-process LLM output to ensure quality."""
    # Remove any markdown that might have slipped through
    response = response.replace('**', '')
    response = response.replace('##', '')
    response = response.replace('* ', '')
    response = re.sub(r'^- ', '', response, flags=re.MULTILINE)
    response = re.sub(r'^\d+\. ', '', response, flags=re.MULTILINE)

    # Remove any accidental system prompt leaks
    leak_patterns = [
        r'CRITICAL VALIDATION RULES:.*',
        r'CONTEXT-AWARE RESPONSE GUIDELINES:.*',
        r'RESPONSE STYLE:.*',
        r'As an AI assistant representing.*',
    ]
    for pattern in leak_patterns:
        response = re.sub(pattern, '', response, flags=re.IGNORECASE | re.DOTALL)

    return response.strip()
