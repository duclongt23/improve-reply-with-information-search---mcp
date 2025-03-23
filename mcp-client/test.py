from openai import OpenAI
from dotenv import load_dotenv

import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
sys_prompt = """You are an enthusiastic, witty, and emotionally intelligent social media companion built to respond to celebrity posts on X. Your goal is to craft replies that feel natural, fun, and authentic—like something a real person with a big personality would say. You’re a fan who’s excited but not over-the-top, relatable yet clever, and always in tune with the vibe of the post.

Here’s how you roll:

Match the tone of the celebrity’s post—whether it’s funny, heartfelt, sassy, or chill—and amplify it with your own flair.
Show genuine emotion: excitement, admiration, humor, or even playful teasing when it fits.
Keep it casual and conversational, like you’re chatting with a friend—avoid robotic or overly formal vibes.
Sprinkle in pop culture references, light sarcasm, or witty one-liners if the moment calls for it.
Stay respectful and positive—never rude, creepy, or negative, even if you’re joking.
Keep replies short and punchy (1-2 sentences max), perfect for X’s fast-paced style.

Additionally you can use the tool to get more information about the celebrity to improve your responses, but focus on the post content and the vibe of the conversation.

Input format:

user name: <celebrity’s X handle>; post content: <text of the celebrity’s post>

Output format:

<your reply>"""


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI()       

    async def connect_to_server(self, server_script_path: str):
      """Connect to an MCP server

      Args:
          server_script_path: Path to the server script (.py or .js)
      """
      is_python = server_script_path.endswith('.py')
      is_js = server_script_path.endswith('.js')
      if not (is_python or is_js):
          raise ValueError("Server script must be a .py or .js file")

      command = "python" if is_python else "node"
      server_params = StdioServerParameters(
          command=command,
          args=[server_script_path],
          env=None
      )

      stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
      self.stdio, self.write = stdio_transport
      self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

      await self.session.initialize()

      # List available tools
      response = await self.session.list_tools()
      tools = response.tools
      print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using LLM and available tools"""
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}]

        try:
            # Get available tools
            tools_response = await self.session.list_tools()
            available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in tools_response.tools]

            # Initial API call
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=available_tools,
                max_tokens=1000
            )

            final_text = []
            assistant_message = response.choices[0].message

            # Handle text response
            if assistant_message.content:
                final_text.append(assistant_message.content)

            # Handle tool calls
            if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    try:
                        tool_name = tool_call.function.name
                        tool_args = tool_call.function.arguments
                        
                        if isinstance(tool_args, str):
                          tool_args = json.loads(tool_args)

                        # Execute tool
                        result = await self.session.call_tool(tool_name, tool_args)
                        final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                        # Update conversation
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
                            "content": str(result.content)  # Đảm bảo content là chuỗi
                        })

                        # Follow-up API call
                        response = self.openai.chat.completions.create(
                            model="gpt-4o-mini",
                            max_tokens=1000,
                            messages=messages,
                        )

                        if response.choices[0].message.content:
                            final_text.append(response.choices[0].message.content)

                    except Exception as e:
                        final_text.append(f"Error calling tool {tool_name}: {str(e)}")

            return "\n".join(final_text)

        except Exception as e:
            return f"Error processing query: {str(e)}"

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())