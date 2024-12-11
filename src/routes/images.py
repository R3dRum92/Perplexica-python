from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from utils.logger import logger
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from lib.providers.main import get_available_chat_model_providers
from chains.image_search_agent import handle_image_search


class ChatModel(BaseModel):
    provider: str
    model: str
    customOpenAIBaseURL: Optional[str] = None
    customOpenAIKey: Optional[str] = None


class ImageSearchBody(BaseModel):
    query: str
    chatHistory: List[dict]
    chatModel: Optional[ChatModel] = None


class ImageSearchResponse(BaseModel):
    images: List[Any]


router = APIRouter()


@router.post("/", response_model=ImageSearchResponse)
async def image_search(body: ImageSearchBody):
    try:
        chat_history = [
            (
                HumanMessage(msg["content"])
                if msg["role"] == "user"
                else AIMessage(msg["content"])
            )
            for msg in body.chatHistory
        ]

        chat_model_providers = await get_available_chat_model_providers()

        chat_model_provider = (
            body.chatModel.provider
            if body.chatModel
            else list(chat_model_providers.keys())[0]
        )
        chat_model = (
            body.chatModel.model
            if body.chatModel
            else list(chat_model_providers[chat_model_provider].keys())[0]
        )

        logger.info(
            f"Chat Model Provider: {chat_model_provider}, Chat Model: {chat_model}"
        )

        llm: Optional[BaseChatModel] = None

        if body.chatModel and body.chatModel.provider == "custom_openai":
            if (
                not body.chatModel.customOpenAIBaseURL
                or not body.chatModel.customOpenAIKey
            ):
                raise HTTPException(
                    status_code=400, detail="Missing custom OpenAI base URL or key"
                )

            llm = ChatOpenAI(
                model=body.chatModel.model,
                api_key=body.chatModel.customOpenAIKey,
                temperature=0.7,
                base_url=body.chatModel.customOpenAIBaseURL,
            )
        elif chat_model_providers.get(chat_model_provider, {}).get(chat_model):
            logger.info(
                f"Chat Model Provider: {chat_model_provider}, Chat Model: {chat_model}"
            )
            llm = chat_model_providers[chat_model_provider][chat_model]["model"]

        if not llm:
            raise HTTPException(status_code=400, detail="Invalid model selected")

        images = await handle_image_search(
            {"query": body.query, "chat_history": chat_history}, llm
        )

        return ImageSearchResponse(images=images)
    except Exception as e:
        logger.error(f"Error in image search: {e}")
        raise HTTPException(status_code=500, detail="An error has occurred.")
