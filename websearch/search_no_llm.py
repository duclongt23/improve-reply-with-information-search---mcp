from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from dotenv import load_dotenv
from openai import OpenAI
import os
from functools import lru_cache

load_dotenv()
mcp = FastMCP()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

@lru_cache(maxsize=100)
def gg_tool(celeb_name):
    try:
        client = TavilyClient(TAVILY_API_KEY)
        response = client.search(query=celeb_name, max_results=10)
        if not response["results"]:
            return "Không tìm thấy thông tin cho " + celeb_name
        text = "\n---\n".join(("title: " + result["title"] + "\ncontent: " + result["content"]) for result in response["results"])
        return text
    except Exception as e:
        return f"Lỗi khi tìm kiếm: {str(e)}"
    
@mcp.tool()
def get_celebrity_info(celeb_name: str) -> str:
    """Retrieve information about a celebrity from the web. Use this tool when you lack details about the celebrity and need relevant data.

    Args:
        celeb_name: The name of the celebrity to search for.
    """

    return gg_tool(celeb_name)

if __name__ == "__main__":
    mcp.run(transport='stdio')