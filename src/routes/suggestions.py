from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from utils.logger import logger
from lib.providers.main import get_available_chat_model_providers
from chains.suggestion_generator_agent import (
    generate_suggestions,
    SuggestionGeneratorInput,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel


router = APIRouter()


class ChatModel(BaseModel):
    provider: str
    model: str
    customOpenAIBaseURL: Optional[str] = None
    customOpenAIKey: Optional[str] = None


class Message(BaseModel):
    role: str
    content: str


class SuggestionsBody(BaseModel):
    chatHistory: List[Message]
    chatModel: Optional[ChatModel] = None


@router.post("/")
async def generate_suggestions_endpoint(body: SuggestionsBody):
    try:
        chat_history = []
        for msg in body.chatHistory:
            if msg.role == "user":
                chat_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                chat_history.append(AIMessage(content=msg.content))
            else:
                chat_history.append(HumanMessage(content=msg.content))

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
                model_name=body.chatModel.model,
                openai_api_key=body.chatModel.customOpenAIKey,
                temperature=0.7,
                configuration={"base_url": body.chatModel.customOpenAIBaseURL},
            )

        elif (
            chat_model_provider in chat_model_providers
            and chat_model in chat_model_providers[chat_model_provider]
        ):
            llm = chat_model_providers[chat_model_provider][chat_model]["model"]

        if not llm:
            raise HTTPException(status_code=400, detail="Invalid model selected")

        suggestions = await generate_suggestions(
            SuggestionGeneratorInput(chat_history=chat_history), llm
        )

        return {"suggestions": suggestions}

    except Exception as e:
        logger.error(f"Error in generating suggestions: {e}")
        raise HTTPException(status_code=500, detail="An error has occurred.")
