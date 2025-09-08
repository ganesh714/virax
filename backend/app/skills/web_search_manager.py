# /app/skills/web_search_manager.py

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from tavily import TavilyClient

# --- Pydantic Model for the request body ---
class SearchQuery(BaseModel):
    query: str

# --- FastAPI Router Setup ---
router = APIRouter(
    prefix="/search",
    tags=["Web Search"]
)

# --- API Endpoint ---

@router.post("/", summary="Perform a web search")
async def perform_web_search(search_query: SearchQuery):
    """
    Takes a user query and returns search results from the Tavily API.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="TAVILY_API_KEY not found in .env file.")
    
    try:
        # Initialize the Tavily client
        tavily = TavilyClient(api_key=api_key)
        
        # Perform the search. You can adjust search_depth for more detailed results.
        response = tavily.search(
            query=search_query.query, 
            search_depth="basic", # 'basic' is faster, 'advanced' is more detailed
            max_results=5
        )
        
        return {"results": response['results']}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred with the search API: {str(e)}")