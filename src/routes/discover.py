from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import random
import asyncio
from utils.logger import logger
from typing import List, Dict, Any
from lib.searxng import search_searxng, SearxngSearchOptions

router = APIRouter()


@router.get("/")
async def discover() -> Dict[str, Any]:
    try:
        search_tasks = [
            search_searxng(
                "site:businessinsider.com AI",
                SearxngSearchOptions(engines=["bing_news"], pageno=1),
            ),
            search_searxng(
                "site:www.exchangewire.com AI",
                SearxngSearchOptions(engines=["bing_news"], pageno=1),
            ),
            search_searxng(
                "site:yahoo.com AI",
                SearxngSearchOptions(engines=["bing_news"], pageno=1),
            ),
            search_searxng(
                "site:businessinsider.com tech",
                SearxngSearchOptions(engines=["bing_news"], pageno=1),
            ),
            search_searxng(
                "site:www.exchangewire.com tech",
                SearxngSearchOptions(engines=["bing_news"], pageno=1),
            ),
            search_searxng(
                "site:yahoo.com tech",
                SearxngSearchOptions(engines=["bing_news"], pageno=1),
            ),
        ]

        results = await asyncio.gather(*search_tasks)

        data = [result["results"] for result in results]
        flattened_data = [item for sublist in data for item in sublist]
        random.shuffle(flattened_data)

        return JSONResponse(content={"blogs": flattened_data})
    except Exception as e:
        logger.error(f"Error in discover route: {str(e)}")
        raise HTTPException(status_code=500, detail="An error has occurred")
