from langchain_anthropic import ChatAnthropic
from config import get_anthropic_api_key
from utils.logger import logger


async def load_anthropic_chat_models() -> dict:
    anthropic_api_key = get_anthropic_api_key()

    if not anthropic_api_key:
        return {}

    try:
        chat_models = {
            "claude-3-5-sonnet-20240620": {
                "displayName": "Claude 3.5 Sonnet",
                "model": ChatAnthropic(
                    temperature=0.7,
                    api_key=anthropic_api_key,
                    model_name="claude-3-5-sonnet-20240620",
                ),
            },
            "claude-3-opus-20240229": {
                "displayName": "Claude 3 Opus",
                "model": ChatAnthropic(
                    temperature=0.7,
                    api_key=anthropic_api_key,
                    model_name="claude-3-opus-20240229",
                ),
            },
            "claude-3-sonnet-20240229": {
                "displayName": "Claude 3 Sonnet",
                "model": ChatAnthropic(
                    temperature=0.7,
                    api_key=anthropic_api_key,
                    model_name="claude-3-sonnet-20240229",
                ),
            },
            "claude-3-haiku-20240307": {
                "displayName": "Claude 3 Haiku",
                "model": ChatAnthropic(
                    temperature=0.7,
                    api_key=anthropic_api_key,
                    model_name="claude-3-haiku-20240307",
                ),
            },
        }

        logger.info("Anthropic chat models loaded successfully.")
        return chat_models

    except Exception as e:
        logger.error(f"Error loading Anthropic models: {e}")
        return {}
