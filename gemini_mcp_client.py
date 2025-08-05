"""
This is a simple MCP client using Gemini API and streamable HTTP.
It’s for testing MCP servers using Google's Gemini model.

Usage:
    python gemini_mcp_client.py http://localhost:5000/mcp
"""

import argparse
import json
import logging
import sys
from contextlib import AsyncExitStack
from functools import partial
from typing import Optional

import anyio
from dotenv import load_dotenv
import google.generativeai as genai

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini_client")

# Load Gemini API key
import os
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

genai.configure(api_key=api_key)


class MCPGeminiClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.history = []

    async def get_tools_schema(self):
        response = await self.session.list_tools()
        tools = [{
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema
        } for tool in response.tools]
        return tools

    def format_prompt(self, query: str, tools: list) -> str:
        prompt = "You are an assistant. Available tools:\n"
        for tool in tools:
            prompt += f"- {tool['name']}: {tool['description']}\n"
        prompt += "\nBased on user's query, you can choose to call a tool.\n"
        prompt += f"User: {query}\n"
        prompt += "Reply with either your answer or a tool call like:\n"
        prompt += "`CALL <tool_name> <json_args>`\n"
        return prompt

    def parse_tool_call(self, response: str):
        if response.startswith("CALL"):
            try:
                _, name, args_str = response.strip().split(" ", 2)
                args = json.loads(args_str)
                return name, args
            except Exception:
                return None
        return None

    async def process_query(self, query: str) -> str:
        tools = await self.get_tools_schema()
        prompt = self.format_prompt(query, tools)

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        result = response.text.strip()
        final_text = []

        tool_call = self.parse_tool_call(result)
        if tool_call:
            tool_name, tool_args = tool_call
            final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

            # Execute the tool
            result_obj = await self.session.call_tool(tool_name, tool_args)
            tool_output = result_obj.content[0].text
            final_text.append(f"[Tool call result: {tool_output}]")

            # Send tool result back to Gemini
            follow_up = model.generate_content([
                f"User asked: {query}",
                f"Tool {tool_name} responded: {tool_output}",
                "Now summarize or continue the reply:"
            ])
            final_text.append(follow_up.text.strip())
        else:
            final_text.append(result)

        return "\n".join(final_text)

    async def chat_loop(self):
        print("\nMCP Gemini Client Started!")
        print("Type your queries, 'clear' to clear history, or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'clear':
                    self.history = []
                    print("History cleared.")
                    continue

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def run_session(read_stream, write_stream):
    client = MCPGeminiClient()
    async with ClientSession(read_stream, write_stream) as session:
        client.session = session
        logger.info("Initializing session")
        await session.initialize()
        logger.info("Initialized")
        await client.chat_loop()


async def main(url: str, args: list[str]):
    async with streamablehttp_client(url) as streams:
        await run_session(*streams[:2])


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL to connect to")
    parser.add_argument("args", nargs="*", help="Additional arguments")
    args = parser.parse_args()
    anyio.run(partial(main, args.url, args.args), backend="trio")


if __name__ == "__main__":
    cli()
