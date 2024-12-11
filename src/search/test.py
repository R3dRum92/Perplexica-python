import pytest
from unittest.mock import MagicMock
from search.meta_search_agent import (
    MetaSearchAgent,
    Document,
)  # Update with your actual import path


@pytest.fixture
def setup_agent():
    # Mock LLM and initialize the agent
    llm_mock = MagicMock()
    llm_mock.invoke = MagicMock(return_value={"content": "summarized content"})

    config = {
        "searchWeb": False,
        "rerank": False,
        "summarizer": True,
        "rerankThreshold": 0.3,
        "queryGeneratorPrompt": "some prompt",
        "responsePrompt": "some response",
        "activeEngines": ["engine1"],
    }
    agent = MetaSearchAgent(config)

    return agent, llm_mock


@pytest.mark.asyncio
async def test_parse_links_with_summarization(setup_agent, mock_get_docs):
    agent, llm_mock = setup_agent

    # Mock the return value of getDocumentsFromLinks
    mock_get_docs.return_value = [
        Document(
            page_content="Document content", metadata={"url": "http://example.com"}
        )
    ]

    doc_groups = [{"pageContent": "Some document content"}]
    question = "Summarize this document"

    # Call the method to test
    summarized_docs = await agent.parse_links(doc_groups, question, llm_mock)

    # Assert that llm.invoke was called
    llm_mock.invoke.assert_called_once()

    # Check the returned result
    assert len(summarized_docs) == 1
    assert "summarized content" in summarized_docs[0].page_content
