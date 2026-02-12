import os
import requests
from typing import Optional
from dotenv import load_dotenv

class WebSearchService:
    """Web search service using Tavily API"""
    
    BASE_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        
        if not self.api_key:
            raise ValueError("Tavily API key is required. Set TAVILY_API_KEY in environment variables or .env file")

    def search(self, query: str, max_results: int = 5):
        """
        Search the web for information
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            dict: {"query": str, "results": [{"title": str, "url": str, "content": str}]} 
                  or {"error": str} on failure
        """
        if not query or not query.strip():
            return {"error": "Search query cannot be empty"}
            
        if max_results < 1 or max_results > 20:
            return {"error": "max_results must be between 1 and 20"}
        
        try:
            payload = {
                "api_key": self.api_key,
                "query": query.strip(),
                "num_results": max_results
            }
            
            response = requests.post(self.BASE_URL, json=payload, timeout=15)
            
            if response.status_code == 401:
                return {"error": "Invalid API key - check your Tavily API credentials"}
            elif response.status_code == 429:
                return {"error": "API rate limit exceeded - try again later"}
            elif not response.ok:
                return {"error": f"Search API error: {response.status_code}"}
            
            data = response.json()
            results = []
            
            for r in data.get("results", []):
                results.append({
                    "title": r.get("title", "No title"),
                    "url": r.get("url", ""),
                    "content": r.get("content", "No content available")
                })

            return {"query": query, "results": results}
            
        except requests.exceptions.Timeout:
            return {"error": "Search request timed out - check your internet connection"}
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to search service - check your internet connection"}
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}