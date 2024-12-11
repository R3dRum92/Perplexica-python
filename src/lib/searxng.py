import requests
from typing import List, Dict, Optional, Any
import logging  # Import logging module
import httpx
from dataclasses import dataclass, field

from config import get_searxng_API_endpoint

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of log messages
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()  # Log to console; you can add FileHandler to log to a file
    ],
)
logger = logging.getLogger(__name__)  # Create a logger instance


@dataclass
class SearxngSearchOptions:
    categories: Optional[List[str]] = field(default_factory=list)
    engines: Optional[List[str]] = field(default_factory=list)
    language: Optional[str] = None
    pageno: Optional[int] = None

    def to_params(self) -> Dict[str, Any]:
        params = {}
        if self.categories:
            params["categories"] = ",".join(self.categories)
        if self.engines:
            params["engines"] = ",".join(self.engines)
        if self.language:
            params["language"] = self.language
        if self.pageno:
            params["pageno"] = str(self.pageno)
        return params


class SearxngSearchResult:
    def __init__(
        self,
        title: str,
        url: str,
        img_src: Optional[str] = None,
        thumbnail_src: Optional[str] = None,
        thumbnail: Optional[str] = None,
        content: Optional[str] = None,
        author: Optional[str] = None,
        iframe_src: Optional[str] = None,
    ):
        self.title = title
        self.url = url
        self.img_src = img_src
        self.thumbnail_src = thumbnail_src
        self.thumbnail = thumbnail
        self.content = content
        self.author = author
        self.iframe_src = iframe_src


async def search_searxng(
    query: str, opts: Optional[SearxngSearchOptions] = None
) -> Dict[str, Any]:
    """
    Searches SearxNG for images based on the rephrased query.

    Args:
        query (str): The rephrased search query.
        opts (Optional[SearxngSearchOptions]): Search options.

    Returns:
        Dict[str, Any]: The JSON response from SearxNG.
    """
    searxng_url = get_searxng_API_endpoint()
    url = f"{searxng_url}/search"

    params = {"q": query, "format": "json"}

    if opts:
        params.update(opts.to_params())

    logger.info(f"Sending request to SearxNG: {url} with params: {params}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            logger.info("Received successful response from SearxNG.")
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise


def display_results(results: List[SearxngSearchResult], suggestions: List = []):
    """
    Display search results in a readable format.

    :param results: List of SearxngSearchResult objects.
    :param suggestions: List of suggestions from SearxNG.
    """
    if not results and not suggestions:
        logger.warning("No results or suggestions to display.")
        return

    if results:
        logger.info(f"Displaying {len(results)} search results:")
        for idx, result in enumerate(results, start=1):
            print(f"\nResult {idx}:")
            print(f"Title: {result.title}")
            print(f"URL: {result.url}")
            if result.content:
                print(f"Description: {result.content}")
            print("-" * 40)

    if suggestions:
        logger.info(f"Displaying {len(suggestions)} suggestions:")
        for idx, suggestion in enumerate(suggestions, start=1):
            print(f"\nSuggestion {idx}: {suggestion}")
        print("-" * 40)


if __name__ == "__main__":
    pass
