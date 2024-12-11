from fastapi import APIRouter, HTTPException
from typing import Dict
from utils.logger import logger
from asyncio import gather
from lib.providers.main import (
    get_available_chat_model_providers,
    get_available_embedding_model_providers,
)

router = APIRouter()


@router.get("/")
async def get_model_providers():
    try:
        chat_model_providers, embedding_model_providers = await gather(
            get_available_chat_model_providers(),
            get_available_embedding_model_providers(),
        )

        for provider in chat_model_providers:
            for model in chat_model_providers[provider]:
                chat_model_providers[provider][model].pop("model", None)

        for provider in embedding_model_providers:
            for model in embedding_model_providers[provider]:
                embedding_model_providers[provider][model].pop("model", None)

        return {
            "chatModelProviders": chat_model_providers,
            "embeddingModelProviders": embedding_model_providers,
        }
    except Exception as e:
        logger.error(f"Error in getting model providers: {e}")
        raise HTTPException(status_code=500, detail="An error has occurred.")
