from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains.llm import LLMChain
from utils.format_history import format_chat_history_as_string
from utils.logger import logger
from lib.searxng import search_searxng, SearxngSearchOptions

image_search_chain_prompt = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question so it is a standalone question that can be used by the LLM to search the web for images.
You need to make sure the rephrased question agrees with the conversation and is relevant to the conversation.

Example:
1. Follow up question: What is a cat?
Rephrased: A cat

2. Follow up question: What is a car? How does it works?
Rephrased: Car working

3. Follow up question: How does an AC work?
Rephrased: AC working

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

prompt_template = PromptTemplate(
    input_variables=["chat_history", "query"], template=image_search_chain_prompt
)


def create_llm_chain(llm: BaseChatModel) -> LLMChain:
    return LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_key="rephrased_query",
    )


str_parser = StrOutputParser()


async def search_images(rephrased_query: str) -> List[Dict[str, str]]:
    search_options = SearxngSearchOptions(engines=["bing images", "google images"])
    res = await search_searxng(rephrased_query, opts=search_options)

    images = []
    for result in res.get("results", []):
        if result.get("img_src") and result.get("url") and result.get("title"):
            images.append(
                {
                    "img_src": result["img_src"],
                    "url": result["url"],
                    "title": result["title"],
                }
            )

    logger.info(f"Found {len(images)} images for query: '{rephrased_query}'")
    return images[:10]


async def handle_image_search(
    input_data: Dict[str, Any], llm: BaseChatModel
) -> List[Dict[str, str]]:
    try:
        formatted_history = format_chat_history_as_string(input_data["chat_history"])
        logger.debug(f"Formatted chat history: {formatted_history}")

        llm_chain = create_llm_chain(llm)

        chain_output = await llm_chain.arun(
            {
                "chat_history": formatted_history,
                "query": input_data["query"],
            }
        )
        logger.debug(f"Chain output: {chain_output}")

        rephrased_query = str_parser.parse(chain_output)
        logger.info(f"Rephrased query: '{rephrased_query}")

        images = await search_images(rephrased_query)

        return images

    except Exception as e:
        logger.error(f"Error in handle_image_search: {e}")
        raise
