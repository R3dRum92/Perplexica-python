from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.logger import logger
import uvicorn
from config import get_port
from typing import List, Dict
from chains.image_search_agent import handle_image_search
from chains.suggestion_generator_agent import (
    generate_suggestions,
    SuggestionGeneratorInput,
)
from chains.video_search_agent import handle_video_search
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from routes.config_route import router as config_router
from routes.discover import router as discover_router
from routes.images import router as image_router
from routes.models import router as model_router
from routes.suggestions import router as suggestion_router
from routes.uploads import router as upload_router
from routes.videos import router as video_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api")
async def get_status():
    logger.info("API status endpoint hit")
    return {"status": "ok"}


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Uncaught exception occured: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"},
    )


app.include_router(config_router, prefix="/api/config", tags=["config"])
app.include_router(discover_router, prefix="/api/discover", tags=["discover"])
app.include_router(image_router, prefix="/api/images", tags=["images"])
app.include_router(model_router, prefix="/api/models", tags=["models"])
app.include_router(suggestion_router, prefix="/api/suggestions", tags=["suggestions"])
app.include_router(upload_router, prefix="/api/uploads", tags=["uploads"])
app.include_router(video_router, prefix="/api/videos", tags=["videos"])


# # Endpoint to handle image search
# @app.post("/search-images")
# async def search_images_endpoint(chat_history: List[Dict[str, str]], query: str):
#     """
#     Endpoint to handle image search based on chat history and a follow-up query.

#     Args:
#         chat_history (List[Dict[str, str]]): List of messages in the chat history.
#         query (str): The follow-up query to rephrase and search for images.

#     Returns:
#         Dict[str, List[Dict[str, str]]]: A dictionary containing a list of image results.
#     """
#     try:
#         # Convert chat history to BaseMessage objects
#         base_messages = []
#         for message in chat_history:
#             role = message.get("role", "user")
#             content = message.get("content", "")
#             if role == "user":
#                 base_messages.append(HumanMessage(content=content))
#             elif role == "assistant":
#                 base_messages.append(AIMessage(content=content))
#             else:
#                 base_messages.append(BaseMessage(content=content))

#         input_data = {"chat_history": base_messages, "query": query}

#         # Initialize your LLM (replace with your actual LLM instance)
#         llm = ChatOpenAI(
#             model="gpt-3.5-turbo",
#             temperature=0.7,
#             api_key="",
#         )

#         # Perform image search
#         images = await handle_image_search(input_data, llm)

#         logger.info(f"Image search completed for query: '{query}'")
#         return {"images": images}

#     except Exception as e:
#         logger.error(f"Error in search_images_endpoint: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error")


# @app.post("/video-search")
# async def video_search(chat_history: List[Dict[str, str]], query: str):
#     try:
#         # Convert chat history to BaseMessage objects
#         base_messages = []
#         for message in chat_history:
#             role = message.get("role", "user")
#             content = message.get("content", "")
#             if role == "user":
#                 base_messages.append(HumanMessage(content=content))
#             elif role == "assistant":
#                 base_messages.append(AIMessage(content=content))
#             else:
#                 base_messages.append(BaseMessage(content=content))

#         input_data = {"chat_history": base_messages, "query": query}

#         # Initialize your LLM (replace with your actual LLM instance)
#         llm = ChatOpenAI(
#             model="gpt-3.5-turbo",
#             temperature=0.7,
#             api_key="",
#         )

#         # Perform image search
#         videos = await handle_video_search(input_data, llm)

#         logger.info(f"Video search completed for query: '{query}'")
#         return {"videos": videos}

#     except Exception as e:
#         logger.error(f"Error in search_videos_endpoint: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error")


# # Endpoint to generate suggestions
# @app.post("/generate-suggestions")
# async def generate_suggestions_endpoint(chat_history: List[Dict[str, str]]):
#     """
#     Endpoint to generate suggestions based on chat history.

#     Args:
#         chat_history (List[Dict[str, str]]): List of messages in the chat history.

#     Returns:
#         Dict[str, List[str]]: A dictionary containing a list of generated suggestions.
#     """
#     try:
#         # Convert chat history to BaseMessage objects
#         base_messages = []
#         for message in chat_history:
#             role = message.get("role", "user")
#             content = message.get("content", "")
#             if role == "user":
#                 base_messages.append(HumanMessage(content=content))
#             elif role == "assistant":
#                 base_messages.append(AIMessage(content=content))
#             else:
#                 # Handle other roles or default to HumanMessage
#                 base_messages.append(HumanMessage(content=content))

#         input_data = SuggestionGeneratorInput(chat_history=base_messages)

#         # Initialize your LLM (replace with your actual LLM instance if different)
#         llm = ChatOpenAI(
#             model="gpt-3.5-turbo",
#             temperature=0.7,
#             api_key="",
#         )

#         # Generate suggestions
#         suggestions = generate_suggestions(input_data, llm)

#         logger.info(f"Generated {len(suggestions)} suggestions.")

#         return {"suggestions": suggestions}

#     except Exception as e:
#         logger.error(f"Error in generate_suggestions_endpoint: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    port = get_port()

    logger.info(f"Server is running on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
