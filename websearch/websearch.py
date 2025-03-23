from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from tavily import TavilyClient
import re
from dotenv import load_dotenv
from openai import OpenAI
import os
from functools import lru_cache

load_dotenv()
mcp = FastMCP()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
client = OpenAI()

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

def preprocess_for_llm(raw_text):
    entries = raw_text.strip().split("\n---\n")
    processed_entries = []
    for entry in entries:
        lines = entry.split("\n")
        if len(lines) >= 2:
            title = lines[0].replace("title: ", "").strip()
            content = " ".join(lines[1:]).replace("content: ", "").strip()
            content = re.sub(r'https?://\S+', '', content)  # Xóa URL
            content = re.sub(r'\[\d+\](?:\[\d+\])*', '', content)  # Xóa chú thích
            content = " ".join(content.split())
            processed_entries.append(f"Title: {title}\nContent: {content}")
    final_text = "\n\n".join(processed_entries)
    return final_text

def celeb_info(celeb_name):
    try:
        completion = client.chat.completions.create(
            model= "gpt-4o-mini",
            messages=[
            {
                "role": "system",
                "content": """You are an expert at extracting concise, relevant information from text. Given the following text about a celebrity, extract only the key details below if possible, only answer the output. Ignore irrelevant or redundant information:
Output should be in the following format:
- Basic Info (name, age, occupation)
- Achievements (notable accomplishments like records, awards, or contributions)
- Recent Events (specific dates and events from 2022 onward)
- Interests/Passions (hobbies or causes they care about)
- Personal Style/Tone (their communication style, e.g., humorous, formal)
"""
            },
            {
                "role": "user",
                "content": preprocess_for_llm(gg_tool(celeb_name))
            }
            ] 
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Lỗi khi gọi LLM: {str(e)}"

@mcp.tool()
def get_celebrity_info(celeb_name: str) -> str:
    """Retrieve information about a celebrity from the web. Use this tool when you lack details about the celebrity and need relevant data.

    Args:
        celeb_name: The name of the celebrity to search for.
    """

    return celeb_info(celeb_name)

# Chạy server
if __name__ == "__main__":
    mcp.run(transport='stdio')