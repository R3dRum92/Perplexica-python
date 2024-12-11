from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains.llm import LLMChain
from utils.format_history import format_chat_history_as_string
from utils.logger import logger
from lib.searxng import search_searxng, SearxngSearchOptions

video_search_chain_prompt = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question so it is a standalone question that can be used by the LLM to search Youtube for videos.
You need to make sure the rephrased question agrees with the conversation and is relevant to the conversation.

Example:
1. Follow up question: How does a car work?
Rephrased: How does a car work?

2. Follow up question: What is the theory of relativity?
Rephrased: What is theory of relativity

3. Follow up question: How does an AC work?
Rephrased: How does an AC work

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

prompt_template = PromptTemplate(
    input_variables=["chat_history", "query"], template=video_search_chain_prompt
)


def create_llm_chain(llm: BaseChatModel) -> LLMChain:
    return LLMChain(
        llm=llm, prompt=prompt_template, output_key="rephrased_query", verbose=True
    )


str_parser = StrOutputParser()


async def search_videos(rephrased_query: str) -> List[Dict[str, str]]:
    search_options = SearxngSearchOptions(engines=["youtube"])
    res = await search_searxng(rephrased_query, opts=search_options)

    videos = []
    for result in res.get("results", []):
        if (
            result.get("thumbnail")
            and result.get("url")
            and result.get("title")
            and result.get("iframe_src")
        ):
            videos.append(
                {
                    "thumbnail": result["thumbnail"],
                    "url": result["url"],
                    "title": result["title"],
                    "iframe_src": result["iframe_src"],
                }
            )

    logger.info(f"Found {len(videos)} videos for query: '{rephrased_query}'")
    return videos[:10]


async def handle_video_search(
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

        videos = await search_videos(rephrased_query=rephrased_query)

        return videos

    except Exception as e:
        logger.error(f"Error in handle_video_search: {e}")
        raise
