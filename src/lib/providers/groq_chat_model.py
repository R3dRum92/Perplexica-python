from langchain_openai import ChatOpenAI
from config import get_groq_api_key
from utils.logger import logger


async def load_groq_chat_models() -> dict:
    groq_api_key = get_groq_api_key()

    if not groq_api_key:
        return {}

    try:
        chat_models = {
            "llama-3.2-3b-preview": {
                "displayName": "Llama 3.2 3B",
                "model": ChatOpenAI(
                    api_key=groq_api_key,
                    model="llama-3.2-3b-preview",
                    temperature=0.7,
                    base_url="https://api.groq.com/openai/v1",
                ),
            },
            "llama-3.2-11b-vision-preview": {
                "displayName": "Llama 3.2 11B Vision",
                "model": ChatOpenAI(
                    api_key=groq_api_key,
                    model="llama-3.2-11b-vision-preview",
                    temperature=0.7,
                    base_url="https://api.groq.com/openai/v1",
                ),
            },
            "llama-3.2-90b-vision-preview": {
                "displayName": "Llama 3.2 90B Vision",
                "model": ChatOpenAI(
                    api_key=groq_api_key,
                    model="llama-3.2-90b-vision-preview",
                    temperature=0.7,
                    base_url="https://api.groq.com/openai/v1",
                ),
            },
            "llama-3.1-70b-versatile": {
                "displayName": "Llama 3.1 70B",
                "model": ChatOpenAI(
                    api_key=groq_api_key,
                    model="llama-3.1-70b-versatile",
                    temperature=0.7,
                    base_url="https://api.groq.com/openai/v1",
                ),
            },
            "llama-3.1-8b-instant": {
                "displayName": "Llama 3.1 8B",
                "model": ChatOpenAI(
                    api_key=groq_api_key,
                    model="llama-3.1-8b-instant",
                    temperature=0.7,
                    base_url="https://api.groq.com/openai/v1",
                ),
            },
            "llama3-8b-8192": {
                "displayName": "LLaMA3 8B",
                "model": ChatOpenAI(
                    api_key=groq_api_key,
                    model="llama3-8b-8192",
                    temperature=0.7,
                    base_url="https://api.groq.com/openai/v1",
                ),
            },
            "llama3-70b-8192": {
                "displayName": "LLaMA3 70B",
                "model": ChatOpenAI(
                    api_key=groq_api_key,
                    model="llama3-70b-8192",
                    temperature=0.7,
                    base_url="https://api.groq.com/openai/v1",
                ),
            },
            "mixtral-8x7b-32768": {
                "displayName": "Mixtral 8x7B",
                "model": ChatOpenAI(
                    api_key=groq_api_key,
                    model="mixtral-8x7b-32768",
                    temperature=0.7,
                    base_url="https://api.groq.com/openai/v1",
                ),
            },
            "gemma-7b-it": {
                "displayName": "Gemma 7B",
                "model": ChatOpenAI(
                    api_key=groq_api_key,
                    model="gemma-7b-it",
                    temperature=0.7,
                    base_url="https://api.groq.com/openai/v1",
                ),
            },
            "gemma2-9b-it": {
                "displayName": "Gemma2 9B",
                "model": ChatOpenAI(
                    api_key=groq_api_key,
                    model="gemma2-9b-it",
                    temperature=0.7,
                    base_url="https://api.groq.com/openai/v1",
                ),
            },
        }

        logger.info("Groq chat models loaded successfully.")
        return chat_models
    except Exception as err:
        logger.error(f"Error loading Groq models: {err}")
        return {}
