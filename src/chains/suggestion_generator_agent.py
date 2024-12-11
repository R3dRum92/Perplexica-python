from typing import List
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains.llm import LLMChain
from lib.output_parsers.list_line_output_parser import LineListOutputParser
from utils.logger import logger
from utils.format_history import format_chat_history_as_string

suggestion_generator_prompt = """
You are an AI suggestion generator for an AI powered search engine. You will be given a conversation below. You need to generate 4-5 suggestions based on the conversation. The suggestion should be relevant to the conversation that can be used by the user to ask the chat model for more information.
You need to make sure the suggestions are relevant to the conversation and are helpful to the user. Keep a note that the user might use these suggestions to ask a chat model for more information. 
Make sure the suggestions are medium in length and are informative and relevant to the conversation.

Provide these suggestions separated by newlines between the XML tags <suggestions> and </suggestions>. For example:

<suggestions>
Tell me more about SpaceX and their recent projects
What is the latest news on SpaceX?
Who is the CEO of SpaceX?
</suggestions>

Conversation:
{chat_history}
"""


class SuggestionGeneratorInput:
    def __init__(self, chat_history: List[BaseMessage]):
        self.chat_history = chat_history


output_parser = LineListOutputParser(key="suggestions")

prompt_template = PromptTemplate(
    input_variables=["chat_history"], template=suggestion_generator_prompt
)


def create_suggestion_generator_chain(llm: BaseChatModel) -> LLMChain:
    return LLMChain(
        llm=llm, prompt=prompt_template, output_parser=output_parser, verbose=True
    )


async def generate_suggestions(
    input_data: SuggestionGeneratorInput, llm: BaseChatModel
) -> List[str]:
    try:
        if hasattr(llm, "temperature"):
            llm.temperature = 0

        formatted_history = format_chat_history_as_string(input_data.chat_history)
        logger.debug(f"Formatted chat history: {formatted_history}")

        suggestion_generator_chain = create_suggestion_generator_chain(llm)

        suggestions = suggestion_generator_chain.run(
            {"chat_history": formatted_history}
        )

        logger.info(f"Generated {len(suggestions)} suggestions.")
        return suggestions

    except Exception as e:
        logger.error(f"Error in generate_suggestions: {e}")
        raise
