from fastapi import APIRouter, HTTPException
from typing import Dict
from lib.providers.main import (
    get_available_chat_model_providers,
    get_available_embedding_model_providers,
)
from config import (
    get_openai_api_key,
    get_ollama_api_endpoint,
    get_anthropic_api_key,
    get_gemini_api_key,
    get_groq_api_key,
    update_config,
)
from utils.logger import logger

router = APIRouter()


@router.get("/")
async def get_config():
    try:
        config = {}

        chat_model_providers = await get_available_chat_model_providers()
        embedding_model_providers = await get_available_embedding_model_providers()

        config["chatModelProviders"] = {}
        config["embeddingModelProviders"] = {}

        for provider, models in chat_model_providers.items():
            config["chatModelProviders"][provider] = [
                {"name": model, "displayName": models[model]["displayName"]}
                for model in models
            ]

        for provider, models in embedding_model_providers.items():
            config["embeddingModelProviders"][provider] = [
                {"name": model, "displayName": models[model]["displayName"]}
                for model in models
            ]

        config["openaiApiKey"] = get_openai_api_key()
        config["ollamaApiUrl"] = get_ollama_api_endpoint()
        config["anthropicApiKey"] = get_anthropic_api_key()
        config["groqApiKey"] = get_groq_api_key()
        config["geminiApiKey"] = get_gemini_api_key()

        return config
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail="An error has occurred.")


@router.post("/")
async def update_config_route(config: Dict):
    try:
        updated_config = {
            "API_KEYS": {
                "OPENAI": config.get("openaiApiKey"),
                "GROQ": config.get("groqApiKey"),
                "ANTHROPIC": config.get("anthropicApiKey"),
                "GEMINI": config.get("geminiApiKey"),
            },
            "API_ENDPOINTS": {
                "OLLAMA": config.get("ollamaApiUrl"),
            },
        }

        # Update the configuration
        update_config(updated_config)

        return {"message": "Config updated"}
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(
            status_code=500, detail="An error has occurred while updating the config."
        )
