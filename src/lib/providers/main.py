import asyncio

from lib.providers.groq_chat_model import load_groq_chat_models
from lib.providers.ollama_chat_model import (
    load_ollama_chat_models,
    load_ollama_embeddings_models,
)
from lib.providers.openai_chat_model import (
    load_openai_chat_models,
    load_openai_embeddings_models,
)
from lib.providers.anthropic_chat_model import load_anthropic_chat_models
from lib.providers.transformers_embeddings import load_transformers_embeddings_models
from lib.providers.gemini_chat_model import (
    load_gemini_chat_models,
    load_gemini_embeddings_models,
)

chat_model_providers = {
    "openai": load_openai_chat_models,
    "groq": load_groq_chat_models,
    "ollama": load_ollama_chat_models,
    "anthropic": load_anthropic_chat_models,
    "gemini": load_gemini_chat_models,
}

embedding_model_providers = {
    "openai": load_openai_embeddings_models,
    "local": load_transformers_embeddings_models,
    "ollama": load_ollama_embeddings_models,
    "gemini": load_gemini_embeddings_models,
}


async def get_available_chat_model_providers():
    models = {}
    for provider, load_func in chat_model_providers.items():
        provider_models = await load_func()
        if provider_models:
            models[provider] = provider_models

    models["custom_openai"] = {}

    return models


async def get_available_embedding_model_providers():
    models = {}

    for provider, load_func in embedding_model_providers.items():
        provider_models = await load_func()
        if provider_models:
            models[provider] = provider_models

    return models


async def main():
    chat_models = await get_available_chat_model_providers()
    embedding_models = await get_available_embedding_model_providers()

    print("Available Chat Models:", chat_models)
    print("Available Embedding Models:", embedding_models)


# Running the main async function
if __name__ == "__main__":
    asyncio.run(main())
