from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from config import get_gemini_api_key
from utils.logger import logger


async def load_gemini_chat_models() -> dict:
    gemini_api_key = get_gemini_api_key()

    if not gemini_api_key:
        return {}

    try:
        chat_models = {
            "gemini-1.5-flash": {
                "displayName": "Gemini 1.5 Flash",
                "model": ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", temperature=0.7, api_key=gemini_api_key
                ),
            },
            "gemini-1.5-flash-8b": {
                "displayName": "Gemini 1.5 Flash 8B",
                "model": ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash-8b", temperature=0.7, api_key=gemini_api_key
                ),
            },
            "gemini-1.5-pro": {
                "displayName": "Gemini 1.5 Pro",
                "model": ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro", temperature=0.7, api_key=gemini_api_key
                ),
            },
        }

        logger.info("Gemini chat models loaded successfully.")
        return chat_models
    except Exception as e:
        logger.error(f"Error loading Gemini chat models: {e}")
        return {}


async def load_gemini_embeddings_models() -> dict:
    gemini_api_key = get_gemini_api_key()

    if not gemini_api_key:
        return {}

    try:
        embedding_models = {
            "text-embedding-004": {
                "displayName": "Text Embedding",
                "model": GoogleGenerativeAIEmbeddings(
                    google_api_key=gemini_api_key, model="text-embedding-004"
                ),
            }
        }

        logger.info("Gemini embeddings models loaded successfully.")
        return embedding_models
    except Exception as e:
        logger.error(f"Error loading Gemini embeddings model: {e}")
        return {}
