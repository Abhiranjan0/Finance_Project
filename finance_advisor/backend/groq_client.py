from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import os

load_dotenv()
from groq import Groq
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# Groq Client (Chat)
# -------------------------------------------------

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")


# -------------------------------------------------
# Chat (LLM) Wrapper
# -------------------------------------------------
def chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 800,
    model: Optional[str] = None,
) -> Any:
    response = client.chat.completions.create(
        model=model or MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response


def chat_completion_text(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 800,
    model: Optional[str] = None,
) -> str:
    response = chat_completion(messages, temperature, max_tokens, model)

    if not response.choices:
        return ""

    return response.choices[0].message.content or ""


# -------------------------------------------------
# Embeddings (LOCAL - replaces Azure)
# -------------------------------------------------

# Load once (important for performance)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(
    texts: List[str],
    model: Optional[str] = None,
) -> List[List[float]]:
    """
    Returns embeddings using local model (no API).
    """
    embeddings = embedding_model.encode(texts)
    return embeddings.tolist()