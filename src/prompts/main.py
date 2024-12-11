from prompts.academic_search import (
    academic_search_retriever_prompt,
    academic_search_response_prompt,
)
from prompts.reddit_search import (
    reddit_search_response_prompt,
    reddit_search_retriever_prompt,
)
from prompts.web_search import web_search_response_prompt, web_search_retriever_prompt
from prompts.wolframalpha import (
    wolframalpha_search_retriever_prompt,
    wolframalpha_search_response_prompt,
)
from prompts.writing_assistant import writing_assistant_prompt
from prompts.youtube_search import (
    youtube_search_response_prompt,
    youtube_search_retriever_prompt,
)

prompts = {
    "webSearchResponsePrompt": web_search_response_prompt,
    "webSearchRetrieverPrompt": web_search_retriever_prompt,
    "academicSearchResponsePrompt": academic_search_response_prompt,
    "academicSearchRetrieverPrompt": academic_search_retriever_prompt,
    "redditSearchResponsePrompt": reddit_search_response_prompt,
    "redditSearchRetrieverPrompt": reddit_search_retriever_prompt,
    "wolframAlphaSearchResponsePrompt": wolframalpha_search_response_prompt,
    "wolframAlphaSearchRetrieverPrompt": wolframalpha_search_retriever_prompt,
    "writingAssistantPrompt": writing_assistant_prompt,
    "youtubeSearchResponsePrompt": youtube_search_response_prompt,
    "youtubeSearchRetrieverPrompt": youtube_search_retriever_prompt,
}
