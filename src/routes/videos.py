from fastapi import APIRouter, HTTPException, Body
from typing import List, Optional
from pydantic import BaseModel
from utils.logger import logger
from lib.providers.main import get_available_chat_model_providers
from chains.video_search_agent import handle_video_search
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatModel(BaseModel):
    provider: str
    model: str
    customOpenAIBaseURL: Optional[str] = None
    customOpenAIKey: Optional[str] = None


class VideoSearchBody(BaseModel):
    query: str
    chatHistory: List[dict]
    chatModel: Optional[ChatModel] = None


router = APIRouter()


@router.post("/", response_model=dict)
async def video_search(body: VideoSearchBody):
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

        llm = None

        if body.chatModel and body.chatModel.provider == "custom_openai":
            if (
                not body.chatModel.customOpenAIBaseURL
                or not body.chatModel.customOpenAIKey
            ):
                raise HTTPException(
                    status_code=400, detail="Missing custom OpenAI base URL or key"
                )
            llm = ChatOpenAI(
                model_name=body.chatModel.model,
                openai_api_key=body.chatModel.customOpenAIKey,
                temperature=0.7,
                base_url=body.chatModel.customOpenAIBaseURL,
            )

        elif chat_model_providers.get(chat_model_provider) and chat_model_providers[
            chat_model_provider
        ].get(chat_model):
            llm = chat_model_providers[chat_model_provider][chat_model]["model"]

        if not llm:
            raise HTTPException(status_code=400, detail="Invalid model selected")

        videos = await handle_video_search(
            {"chat_history": chat_history, "query": body.query}, llm
        )

        return {"videos": videos}

    except Exception as e:
        logger.error(f"Error in video search: {str(e)}")
        raise HTTPException(status_code=500, detail="An error has occurred")
