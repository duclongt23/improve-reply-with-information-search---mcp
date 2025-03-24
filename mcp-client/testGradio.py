import asyncio
from typing import Optional, Dict, List, Any
import json
import os
import logging
import gradio as gr

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration class (unchanged)
class Configuration:
    def __init__(self) -> None:
        self.load_env()

    @staticmethod
    def load_env() -> None:
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            return json.load(f)

# Server class (simplified for prototype)
class Server:
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name = name
        self.config = config
        self.session = None
        self.capabilities = None

    async def initialize(self) -> None:
        logging.info(f"Initializing server: {self.name}")
        self.session = True  # Mock session for demo
        self.capabilities = {'progress': True}

    async def list_tools(self) -> List[Any]:
        return [Tool("analyze_post", "Analyze a post", {"properties": {"content": {"description": "Post text"}}})]

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        return f"Mock result for {tool_name}: {arguments}"

    async def cleanup(self) -> None:
        logging.info(f"Cleaning up server: {self.name}")
        self.session = None

# Tool class (unchanged)
class Tool:
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def format_for_llm(self) -> str:
        args_desc = []
        if 'properties' in self.input_schema:
            for param_name, param_info in self.input_schema['properties'].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                args_desc.append(arg_desc)
        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""

# LLMClient class (unchanged)
class LLMClient:
    def __init__(self) -> None:
        self.openai = OpenAI()

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in LLM request: {e}")
            return f"Error: {str(e)}"

# ChatSession class (modified to return tool call info)
class ChatSession:
    def __init__(self, servers: List[Server], llm_client: LLMClient) -> None:
        self.servers = servers
        self.llm_client = llm_client

    async def initialize_servers(self) -> None:
        for server in self.servers:
            await server.initialize()

    async def process_llm_response(self, llm_response: str) -> tuple[str, bool, str]:
        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        result = await server.execute_tool(tool_call["tool"], tool_call["arguments"])
                        return result, True, llm_response  # Result, has_tool_call, tool_call_json
                return f"No server found with tool: {tool_call['tool']}", True, llm_response
            return llm_response, False, ""  # No tool call
        except json.JSONDecodeError:
            return llm_response, False, ""

    async def generate_reply(self, user_name: str, post_content: str) -> tuple[str, str, str]:
        await self.initialize_servers()
        all_tools = []
        for server in self.servers:
            tools = await server.list_tools()
            all_tools.extend(tools)
        
        tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

        system_message = f"""You are an enthusiastic, witty, and emotionally intelligent social media companion built to respond to celebrity posts on X. Your goal is to craft replies that feel natural, fun, and authentic—like something a real person with a big personality would say. You’re a fan who’s excited but not over-the-top, relatable yet clever, and always in tune with the vibe of the post. To enhance your ability you have access to these tools: 

{tools_description}
Choose the appropriate tool based on the celebrity you reply and the post content. If no tool is needed, reply directly.

IMPORTANT: When you need to use a tool, you must ONLY respond with the exact JSON object format below, nothing else:
{{
    "tool": "tool-name",
    "arguments": {{
        "argument-name": "value"
    }}
}}

After receiving a tool's response:
1. Transform the raw data into a natural, conversational response
2. Keep responses concise but informative
3. Focus on the most relevant information
4. Use appropriate context from the user's question
5. Avoid simply repeating the raw data

Input format:
User name: <celebrity’s X handle>
Post content: <text of the celebrity’s post>

Output format:
<your reply>"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"User name: {user_name}\nPost content: {post_content}"}
        ]

        llm_response = self.llm_client.get_response(messages)
        result, has_tool_call, tool_call_json = await self.process_llm_response(llm_response)
        
        if has_tool_call:  # Tool was used, get final response
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "system", "content": result})
            final_response = self.llm_client.get_response(messages)
            return final_response, "Yes", result
        return result, "No", ""

    async def cleanup_servers(self) -> None:
        for server in self.servers:
            await server.cleanup()

# Gradio interface
async def run_chat(user_name: str, post_content: str):
    config = Configuration()
    server_config = {"mcpServers": {"demo": {"command": "npx", "args": [], "env": {}}}}
    servers = [Server(name, srv_config) for name, srv_config in server_config["mcpServers"].items()]
    llm_client = LLMClient()
    chat_session = ChatSession(servers, llm_client)
    
    try:
        reply, has_tool_call, tool_result = await chat_session.generate_reply(user_name, post_content)
        return reply, has_tool_call, tool_result
    finally:
        await chat_session.cleanup_servers()

def gradio_interface(user_name, post_content):
    reply, has_tool_call, tool_result = asyncio.run(run_chat(user_name, post_content))
    return reply, has_tool_call, tool_result

# Launch Gradio app
with gr.Blocks(title="Celebrity Post Reply Generator") as demo:
    gr.Markdown("# Celebrity Post Reply Generator")
    gr.Markdown("Enter a celebrity's X handle and their post content to generate a witty reply!")
    
    with gr.Row():
        user_name_input = gr.Textbox(label="Celebrity X Handle", placeholder="e.g., @elonmusk")
        post_content_input = gr.Textbox(label="Post Content", placeholder="e.g., Just launched a new rocket!")
    
    submit_btn = gr.Button("Generate Reply")
    
    with gr.Column():
        output_reply = gr.Textbox(label="Generated Reply", interactive=False)
        output_tool_call = gr.Textbox(label="Tool Call Used?", interactive=False)
        output_tool_result = gr.Textbox(label="Tool Call Result (if any)", interactive=False)
    
    submit_btn.click(
        fn=gradio_interface,
        inputs=[user_name_input, post_content_input],
        outputs=[output_reply, output_tool_call, output_tool_result]
    )

demo.launch()