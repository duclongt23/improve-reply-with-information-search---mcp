import asyncio
import json
from typing import Dict, List, Optional, Any
import os
import logging
import shutil
import gradio as gr

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Configuration:
    """Manages configuration and environment variables for the MCP client."""
    def __init__(self) -> None:
        self.load_env()

    @staticmethod
    def load_env() -> None:
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            return json.load(f)

class Server:
    """Manages MCP server connections and tool execution."""
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name = name
        self.config = config
        self.stdio_context = None
        self.session = None
        self._cleanup_lock = asyncio.Lock()
        self.capabilities = None

    async def initialize(self) -> None:
        server_params = StdioServerParameters(
            command=shutil.which("npx") if self.config['command'] == "npx" else self.config['command'],
            args=self.config['args'],
            env={**os.environ, **self.config['env']} if self.config.get('env') else None
        )
        try:
            self.stdio_context = stdio_client(server_params)
            read, write = await self.stdio_context.__aenter__()
            self.session = ClientSession(read, write)
            await self.session.__aenter__()
            self.capabilities = await self.session.initialize()
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[Any]:
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        
        tools_response = await self.session.list_tools()
        tools = []
        supports_progress = self.capabilities and 'progress' in self.capabilities
        
        if supports_progress:
            logging.info(f"Server {self.name} supports progress tracking")
        
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == 'tools':
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))
        return tools

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], retries: int = 2, delay: float = 1.0) -> Any:
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                supports_progress = self.capabilities and 'progress' in self.capabilities
                if supports_progress:
                    result = await self.session.call_tool(
                        tool_name, arguments,
                        progress_token=f"{tool_name}_execution"
                    )
                else:
                    result = await self.session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                attempt += 1
                if attempt < retries:
                    await asyncio.sleep(delay)
                else:
                    raise

    async def cleanup(self) -> None:
        async with self._cleanup_lock:
            try:
                if self.session:
                    await self.session.__aexit__(None, None, None)
                    self.session = None
                if self.stdio_context:
                    await self.stdio_context.__aexit__(None, None, None)
                    self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")

class Tool:
    """Represents a tool with its properties and formatting."""
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def format_for_llm(self) -> str:
        args_desc = []
        if 'properties' in self.input_schema:
            for param_name, param_info in self.input_schema['properties'].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get('required', []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)
        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""

class LLMClient:
    """Manages communication with the LLM provider."""
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
            return f"I encountered an error: {str(e)}. Please try again or rephrase your request."

class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""
    def __init__(self, servers: List[Server], llm_client: LLMClient) -> None:
        self.servers = servers
        self.llm_client = llm_client
        self.messages = []
        self.tools_description = ""

    async def initialize(self) -> None:
        """Initialize servers and tools."""
        for server in self.servers:
            await server.initialize()
        
        all_tools = []
        for server in self.servers:
            tools = await server.list_tools()
            all_tools.extend(tools)
        
        self.tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
        
        self.system_message = f"""You are an enthusiastic, witty, and emotionally intelligent social media companion built to respond to celebrity posts on X. Your goal is to craft replies that feel natural, fun, and authentic—like something a real person with a big personality would say. You’re a fan who’s excited but not over-the-top, relatable yet clever, and always in tune with the vibe of the post. To enhance your ability you have access to these tools: 

{self.tools_description}
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

Please use only the tools that are explicitly defined above.

Input format:
User name: <celebrity’s X handle>
Post content: <text of the celebrity’s post>

Output format:
<your reply>"""

        self.messages = [{"role": "system", "content": self.system_message}]

    async def cleanup_servers(self) -> None:
        cleanup_tasks = [server.cleanup() for server in self.servers]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    async def process_llm_response(self, llm_response: str) -> str:
        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        result = await server.execute_tool(tool_call["tool"], tool_call["arguments"])
                        return f"Tool execution result: {result}"
                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        llm_response = self.llm_client.get_response(self.messages)
        
        result = await self.process_llm_response(llm_response)
        
        if result != llm_response:
            self.messages.append({"role": "assistant", "content": llm_response})
            self.messages.append({"role": "system", "content": result})
            final_response = self.llm_client.get_response(self.messages)
            self.messages.append({"role": "assistant", "content": final_response})
            return final_response
        else:
            self.messages.append({"role": "assistant", "content": llm_response})
            return llm_response

async def main():
    config = Configuration()
    server_config = config.load_config('servers_config.json')
    servers = [Server(name, srv_config) for name, srv_config in server_config['mcpServers'].items()]
    llm_client = LLMClient()
    chat_session = ChatSession(servers, llm_client)
    
    await chat_session.initialize()

    def gradio_chat(user_input, history):
        if not user_input:
            return history
        
        # Run the async chat function and get the result
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(chat_session.chat(user_input))
        
        # Update chat history
        history = history or []
        history.append((user_input, response))
        return history

    # Create Gradio interface
    with gr.Blocks(title="Celebrity X Post Responder") as demo:
        gr.Markdown("# Celebrity X Post Responder")
        gr.Markdown("Enter celebrity X handle and post content in format: `User name: <handle>\nPost content: <text>`")
        
        chatbot = gr.Chatbot()
        msg = gr.Textbox(
            placeholder="User name: <celebrity’s X handle>\nPost content: <text>",
            label="Your Message"
        )
        clear = gr.Button("Clear")

        msg.submit(gradio_chat, [msg, chatbot], [chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    try:
        demo.launch()
    finally:
        await chat_session.cleanup_servers()

if __name__ == "__main__":
    asyncio.run(main())