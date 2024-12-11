import requests
from config import get_ollama_api_endpoint, get_keep_alive
from utils.logger import logger
from langchain_ollama import ChatOllama, OllamaEmbeddings


async def load_ollama_chat_models() -> dict:
    ollama_endpoint = get_ollama_api_endpoint()
    keep_alive = get_keep_alive()

    if not ollama_endpoint:
        return {}

    try:
        response = requests.get(
            f"{ollama_endpoint}/api/tags", headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        ollama_models = response.json().get("models", [])

        chat_models = {}
        for model in ollama_models:
            chat_models[model["model"]] = {
                "displayName": model["name"],
                "model": ChatOllama(
                    base_url=ollama_endpoint,
                    model=model["model"],
                    temperature=0.7,
                    keep_alive=keep_alive,
                ),
            }

        logger.info("Ollama chat models loaded successfully.")
        return chat_models

    except Exception as e:
        logger.error(f"Error loading Ollama chat models: {e}")
        return {}


async def load_ollama_embeddings_models() -> dict:
    ollama_endpoint = get_ollama_api_endpoint()

    if not ollama_endpoint:
        return {}

    try:
        response = requests.get(
            f"{ollama_endpoint}/api/tags", headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        ollama_models = response.json().get("models", [])

        embeddings_models = {}
        for model in ollama_models:
            embeddings_models[model["model"]] = {
                "displayName": model["name"],
                "model": OllamaEmbeddings(
                    base_url=ollama_endpoint, model=model["model"]
                ),
            }

        logger.info("Ollama embeddings models loaded successfully.")
        return embeddings_models

    except Exception as e:
        logger.error(f"Error loading Ollama embeddings models: {e}")
        return {}
