import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# Your system prompt remains the same
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
User name: <celebrity’s X handle>
Post content: <text of the celebrity’s post>

Output format:
<your reply>"""

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI()
        self.connected = False

    async def connect_to_server(self, server_script_path: str):
        try:
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
            self.connected = True
            response = await self.session.list_tools()
            return f"Connected successfully! Available tools: {[tool.name for tool in response.tools]}"
        except Exception as e:
            return f"Connection failed: {str(e)}"

    async def process_query(self, query: str) -> str:
        if not self.connected:
            return "Please connect to a server first!"
            
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}]

        try:
            tools_response = await self.session.list_tools()
            available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in tools_response.tools]

            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=available_tools,
                max_tokens=1000
            )

            final_text = []
            assistant_message = response.choices[0].message

            if assistant_message.content:
                final_text.append(assistant_message.content)

            if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
                    
                    result = await self.session.call_tool(tool_name, tool_args)
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

            return "\n----\n".join(final_text)

        except Exception as e:
            return f"Error processing query: {str(e)}"

    async def cleanup(self):
        await self.exit_stack.aclose()

# Gradio interface
async def create_gradio_interface():
    client = MCPClient()
    
    async def connect_fn(server_path):
        result = await client.connect_to_server(server_path)
        return result

    async def process_fn(celebrity_handle, post_content):
        query = f"User name: {celebrity_handle}\nPost content: {post_content}"
        result = await client.process_query(query)
        return result

    with gr.Blocks(title="X Post Responder") as demo:
        gr.Markdown("# X Post Responder")
        gr.Markdown("Connect to server and generate witty responses to celebrity X posts!")
        
        with gr.Row():
            server_input = gr.Textbox(label="Server Script Path", placeholder="Enter path to .py or .js server script")
            connect_btn = gr.Button("Connect")
            status_output = gr.Textbox(label="Connection Status")
        
        with gr.Row():
            with gr.Column():
                celeb_input = gr.Textbox(label="Celebrity X Handle", placeholder="username")
                post_input = gr.Textbox(label="Post Content", placeholder="What's on their mind?")
                submit_btn = gr.Button("Generate Response")
            response_output = gr.Textbox(label="Generated Response")

        connect_btn.click(
            fn=connect_fn,
            inputs=server_input,
            outputs=status_output
        )
        
        submit_btn.click(
            fn=process_fn,
            inputs=[celeb_input, post_input],
            outputs=response_output
        )

    return demo

if __name__ == "__main__":
    demo = asyncio.run(create_gradio_interface())
    demo.launch()