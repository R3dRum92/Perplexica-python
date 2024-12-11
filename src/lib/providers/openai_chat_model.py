from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import get_openai_api_key
from utils.logger import logger


async def load_openai_chat_models() -> dict:
    openai_api_key = get_openai_api_key()

    if not openai_api_key:
        return {}

    try:
        chat_models = {
            "gpt-3.5-turbo": {
                "displayName": "GPT-3.5 Turbo",
                "model": ChatOpenAI(
                    api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.7
                ),
            },
            "gpt-4": {
                "displayName": "GPT-4",
                "model": ChatOpenAI(
                    api_key=openai_api_key, model="gpt-4", temperature=0.7
                ),
            },
            "gpt-4-turbo": {
                "displayName": "GPT-4 Turbo",
                "model": ChatOpenAI(
                    api_key=openai_api_key, model="gpt-4-turbo", temperature=0.7
                ),
            },
            "gpt-4o": {
                "displayName": "GPT-4 omni",
                "model": ChatOpenAI(
                    api_key=openai_api_key, model="gpt-4o", temperature=0.7
                ),
            },
            "gpt-4o-mini": {
                "displayName": "GPT-4 omni mini",
                "model": ChatOpenAI(
                    api_key=openai_api_key, model="gpt-4o-mini", temperature=0.7
                ),
            },
        }

        logger.info("OpenAI chat models loaded successfully.")
        return chat_models

    except Exception as e:
        logger.error(f"Error loading OpenAI models: {e}")
        return {}


async def load_openai_embeddings_models() -> dict:
    openai_api_key = get_openai_api_key()

    if not openai_api_key:
        return {}

    try:
        embedding_models = {
            "text-embedding-3-small": {
                "displayName": "Text Embedding 3 Small",
                "model": OpenAIEmbeddings(
                    api_key=openai_api_key,
                    model="text-embedding-3-small",
                ),
            },
            "text-embedding-3-large": {
                "displayName": "Text Embedding 3 Large",
                "model": OpenAIEmbeddings(
                    api_key=openai_api_key,
                    model="text-embedding-3-large",
                ),
            },
        }

        logger.info("OpenAI embeddings models loaded successfully.")
        return embedding_models
    except Exception as e:
        logger.error(f"Error loading OpenAI embeddings models: {e}")
        return {}


# if __name__ == "__main__":
#     chat_models = load_openai_chat_models()
#     embeddings_models = load_openai_embeddings_models()

#     if not chat_models:
#         print("No chat models loaded. Please check the logs for errors.")

#     else:
#         if not embeddings_models:
#             print("No embeddings models loaded. Please check the logs for errors.")

#         else:

#             # Display loaded chat models
#             print("Loaded Chat Models:")
#             for model_key, model_info in chat_models.items():
#                 display_name = model_info["displayName"]
#                 print(f" - {display_name} ({model_key})")

#             # Display loaded embeddings models
#             print("\nLoaded Embeddings Models:")
#             for model_key, model_info in embeddings_models.items():
#                 display_name = model_info["displayName"]
#                 print(f" - {display_name} ({model_key})")
