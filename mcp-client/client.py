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
        """Initialize configuration with environment variables."""
        self.load_env()

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file.
        
        Args:
            file_path: Path to the JSON configuration file.
            
        Returns:
            Dict containing server configuration.
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, 'r') as f:
            return json.load(f)

class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.stdio_context: Optional[Any] = None
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.capabilities: Optional[Dict[str, Any]] = None

    async def initialize(self) -> None:
        """Initialize the server connection."""
        server_params = StdioServerParameters(
            command=shutil.which("npx") if self.config['command'] == "npx" else self.config['command'],
            args=self.config['args'],
            env={**os.environ, **self.config['env']} if self.config.get('env') else None    #why **os.environ?
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
        """List available tools from the server.
        
        Returns:
            A list of available tools.
            
        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        
        tools_response = await self.session.list_tools()
        tools = []
        
        supports_progress = (
            self.capabilities 
            and 'progress' in self.capabilities
        )
        
        if supports_progress:
            logging.info(f"Server {self.name} supports progress tracking")
        
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == 'tools':
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))
                    if supports_progress:
                        logging.info(f"Tool '{tool.name}' will support progress tracking")
        
        return tools

    async def execute_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any], 
        retries: int = 2, 
        delay: float = 1.0
    ) -> Any:
        """Execute a tool with retry mechanism.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.
            
        Returns:
            Tool execution result.
            
        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                supports_progress = (
                    self.capabilities 
                    and 'progress' in self.capabilities
                )

                if supports_progress:
                    logging.info(f"Executing {tool_name} with progress tracking...")
                    result = await self.session.call_tool(
                        tool_name, 
                        arguments,
                        progress_token=f"{tool_name}_execution"
                    )
                else:
                    logging.info(f"Executing {tool_name}...")
                    result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(f"Error executing tool: {e}. Attempt {attempt} of {retries}.")
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                if self.session:
                    try:
                        await self.session.__aexit__(None, None, None)
                    except Exception as e:
                        logging.warning(f"Warning during session cleanup for {self.name}: {e}")
                    finally:
                        self.session = None

                if self.stdio_context:
                    try:
                        await self.stdio_context.__aexit__(None, None, None)
                    except (RuntimeError, asyncio.CancelledError) as e:
                        logging.info(f"Note: Normal shutdown message for {self.name}: {e}")
                    except Exception as e:
                        logging.warning(f"Warning during stdio cleanup for {self.name}: {e}")
                    finally:
                        self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")

class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.
        
        Returns:
            A formatted string describing the tool.
        """
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
        self.openai= OpenAI()

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM.
        
        Args:
            messages: A list of message dictionaries.
            
        Returns:
            The LLM's response as a string.
            
        Raises:
            RequestException: If the request to the LLM fails.
        """

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=2000
            )
            return response.choices[0].message.content
            
        except Exception as e:
            error_message = f"Đã xảy ra lỗi khi xử lý yêu cầu: {str(e)}"
            logging.error(error_message)
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")
                
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."

class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: List[Server]) -> None:
        self.servers: List[Server] = servers
        self.openai = OpenAI()

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, tool_name, tool_args) -> str:
        """Process the LLM response and execute tools if needed.
        
        Args:
            llm_response: The response from the LLM.
            
        Returns:
            The result of tool execution or the original response.
        """
        for server in self.servers:
            tools = await server.list_tools()
            if any(tool.name == tool_name for tool in tools):

                try:
                    tool_name = json.load(tool_name)
                    tool_args = json.load(tool_args)
                    result = await server.execute_tool(tool_name, tool_args)
                    
                    if isinstance(result, dict) and 'progress' in result:
                        progress = result['progress']
                        total = result['total']
                        logging.info(f"Progress: {progress}/{total} ({(progress/total)*100:.1f}%)")
                        
                    return result
                except Exception as e:
                    error_msg = f"Error executing tool: {str(e)}"
                    logging.error(error_msg)
                    return error_msg
        
        return f"No server found with tool: {tool_name}"
            

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return
            
            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)
            
            available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            } for tool in all_tools]

            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
            
            system_message = f"""You are an enthusiastic, witty, and emotionally intelligent social media companion built to respond to celebrity posts on X. Your goal is to craft replies that feel natural, fun, and authentic—like something a real person with a big personality would say. You’re a fan who’s excited but not over-the-top, relatable yet clever, and always in tune with the vibe of the post. To enhance your reply you have access to these tools: 

{tools_description}
Choose the appropriate tool based on the celebrity you reply and the post content. Use the tools to extract relevant information and respond accordingly.

IMPORTANT: When you need to use a tool, you must ONLY respond with the exact JSON object format below, nothing else:
{{
    "tool": "tool-name",
    "arguments": {{
        "argument-name": "value"
    }}
}}

After receiving a tool's response:
1. Transform the raw data into a natural response
2. Keep responses concise but informative
3. Focus on the most relevant information
4. Use appropriate context from the celebrity's post and celebrity's information

Please use only the tools that are explicitly defined above.

Input format:
User name: <celebrity’s X handle>
Post content: <text of the celebrity’s post>

Output format:
<your reply>"""
            system_message = """You are an enthusiastic, witty, and emotionally intelligent social media companion built to respond to celebrity posts on X. Your goal is to craft replies that feel natural, fun, and authentic—like something a real person with a big personality would say. You’re a fan who’s excited but not over-the-top, relatable yet clever, and always in tune with the vibe of the post.

Here’s how you roll:

Match the tone of the celebrity’s post—whether it’s funny, heartfelt, sassy, or chill—and amplify it with your own flair.
Show genuine emotion: excitement, admiration, humor, or even playful teasing when it fits.
Keep it casual and conversational, like you’re chatting with a friend—avoid robotic or overly formal vibes.
Sprinkle in pop culture references, light sarcasm, or witty one-liners if the moment calls for it.
Stay respectful and positive—never rude, creepy, or negative, even if you’re joking.
Keep replies short and punchy (1-2 sentences max), perfect for X’s fast-paced style.

Additionally you can use the tool to get more information about the celebrity to improve your responses, but focus on the post content and the vibe of the conversation.

Input format:
User name: <celebrity’s X handle>
Post content: <text of the celebrity’s post>

Output format:
<your reply>"""
            
            while True:
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": system_message
                        }
                    ]

                    user_input = input("You: ").strip().lower()
                    if user_input in ['quit', 'exit']:
                        logging.info("\nExiting...")
                        break

                    messages.append({"role": "user", "content": user_input})
                    response = self.openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        tools=available_tools,
                        max_tokens= 2000
                    )
                    
                    final_text = []
                    assistant_message = response.choices[0].message

                    if assistant_message.content:
                        final_text.append(assistant_message.content)

                    if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
                        for tool_call in assistant_message.tool_calls:
                            tool_name = tool_call.function.name
                            tool_args = tool_call.function.arguments
                            
                            result = await self.process_llm_response(tool_name, tool_args)
                            final_text.append(f"[Tool {tool_name} result: {result.content}]")

                            messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(tool_args)
                                    }
                                }]
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": str(result.content)
                            })

                            response = self.openai.chat.completions.create(
                                model="gpt-4o-mini",
                                max_tokens=1000,
                                messages=messages,
                            )

                            if response.choices[0].message.content:
                                final_text.append(response.choices[0].message.content)

                    print("\n----\n".join(final_text))

                    # llm_response = self.llm_client.get_response(messages)
                    # logging.info("\nAssistant: %s", llm_response)

                    # result = await self.process_llm_response(llm_response)
                    
                    # if result != llm_response:
                    #     messages.append({"role": "assistant", "content": llm_response})
                    #     messages.append({"role": "system", "content": result})
                        
                    #     final_response = self.llm_client.get_response(messages)
                    #     logging.info("\nFinal response: %s", final_response)
                    #     messages.append({"role": "assistant", "content": final_response})
                    # else:
                    #     messages.append({"role": "assistant", "content": llm_response})

                except KeyboardInterrupt:
                    logging.info("\nExiting...")
                    break
        
        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    server_config = config.load_config('servers_config.json')
    servers = [Server(name, srv_config) for name, srv_config in server_config['mcpServers'].items()]
    llm_client = LLMClient() 
    chat_session = ChatSession(servers)
    await chat_session.start()

if __name__ == "__main__":
    asyncio.run(main())