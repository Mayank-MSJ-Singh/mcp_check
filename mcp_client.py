"""
Multi-Server MCP Client (ULTRA X-RAY EDITION)
Full lifecycle logs. Full data logs. Timing logs.
If anything breaks, you’ll see it.


find all my events today and sent it to mayank.msj.singh@gmail.com, if no event just sent that have no event today
"""

import json
import logging
import sys
import time
import tracemalloc
import anyio
from contextlib import AsyncExitStack
from typing import Dict

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import OpenAI

# -------------------------------
# HARD-CODED SERVERS
# -------------------------------
KLAVIS_SERVERS = {
    "gmail": "https://gmail-mcp-server.klavis.ai/mcp/?instance_id=e0ddd5ee-45fc-4791-b9f7-5a04d8a58463",
    "github": "https://strata.klavis.ai/mcp/?instance_id=df9ad3af-9eb3-4287-b2f4-acbaa5db1138",
    "linear": "https://linear-mcp-server.klavis.ai/mcp/?instance_id=8e711cd1-909a-4641-95e7-b3d5ee358110",
    "gcalendar": "https://gcalendar-mcp-server.klavis.ai/mcp/?instance_id=9d9a4b34-d0c5-4b8e-b633-1aa101f57de6",
    "gdrive": "https://gdrive-mcp-server.klavis.ai/mcp/?instance_id=ab124495-a682-407e-b9a2-d82bb8ab77d0",
    "jira": "https://strata.klavis.ai/mcp/?instance_id=1c92c41b-f007-4c4b-81ab-be0a6be9b0aa",
    "notion": "https://strata.klavis.ai/mcp/?instance_id=35468960-6bec-4581-b141-dccd41e87742",
    "slack" : "https://slack-mcp-server.klavis.ai/mcp/?instance_id=bfb2bca1-e73b-4c9d-9338-de1ecd35f4ea",
    "attio" : "https://attio-mcp-server.klavis.ai/mcp/?instance_id=82c94b49-2981-4ab2-9433-1f081a36f22c",
    "hackerNews" : "https://hacker-news-mcp-server.klavis.ai/mcp/?instance_id=000c2de5-7296-4417-97e2-6082e77a0050",
}

load_dotenv()

# -------------------------------
# LOGGING SETUP
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("X-RAY-MCP")

tracemalloc.start()


def memory_usage():
    current, peak = tracemalloc.get_traced_memory()
    return f"{current/1024/1024:.3f}MB / peak {peak/1024/1024:.3f}MB"


class MultiMCPClient:
    def __init__(self):
        logger.info("🧩 Creating MultiMCPClient instance")
        self.sessions: Dict[str, ClientSession] = {}
        self.openai = OpenAI()
        self.messages = []
        logger.info(f"📦 Memory on init: {memory_usage()}")

    async def process_query(self, query: str) -> str:
        print("\n" + "=" * 80)
        logger.info(f"🚀 NEW QUERY: {query}")
        logger.info(f"📦 Memory at query start: {memory_usage()}")

        t0 = time.time()

        self.messages.append({"role": "user", "content": query})
        logger.info("📨 User message appended to history")
        logger.info(f"🧠 History length: {len(self.messages)}")

        # ------------------------------------------------------
        # 1. AGGREGATE TOOLS
        # ------------------------------------------------------
        logger.info("🔍 Aggregating tools from all servers...")
        tools_start = time.time()
        all_tools = []

        for service_name, session in self.sessions.items():
            logger.info(f"➡️ Fetching tools from {service_name}...")
            try:
                response = await session.list_tools()
                logger.info(f"📦 {service_name}: {len(response.tools)} tools found")

                for tool in response.tools:
                    logger.info(f"🔧 TOOL: {service_name}.{tool.name}")
                    logger.info(f"📝 Schema: {tool.inputSchema}")

                    prefixed = f"{service_name}_{tool.name}"
                    all_tools.append({
                        "type": "function",
                        "function": {
                            "name": prefixed,
                            "description": f"[{service_name.upper()}] {tool.description}",
                            "parameters": tool.inputSchema
                        }
                    })

            except Exception as e:
                logger.error(f"❌ Failed to fetch tools from {service_name}: {e}")

        logger.info(f"📦 TOTAL TOOLS: {len(all_tools)}")
        logger.info(f"⏱ Tool aggregation took {time.time() - tools_start:.4f}s")

        # ------------------------------------------------------
        # 2. FIRST OPENAI CALL
        # ------------------------------------------------------
        logger.info("🧠 Making first OpenAI call")
        logger.info(f"📨 Sending {len(self.messages)} messages")
        logger.info(f"🔧 Tools attached: {len(all_tools)}")

        ai_start = time.time()
        response = self.openai.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            max_tokens=1000,
            messages=self.messages,
            tools=all_tools if all_tools else None
        )
        logger.info(f"⏱ OpenAI call took {time.time() - ai_start:.4f}s")

        message = response.choices[0].message

        logger.info("📡 RAW AI RESPONSE:")
        logger.info(json.dumps(response.to_dict(), indent=2))

        # ------------------------------------------------------
        # 3. AGENTIC LOOP
        # ------------------------------------------------------
        loop_index = 0

        while message.tool_calls:
            loop_index += 1

            print(f"\n--- 🔄 AGENTIC LOOP ROUND {loop_index} ---")
            logger.info(f"🔄 Loop {loop_index} started")
            logger.info(f"🧰 AI called {len(message.tool_calls)} tool(s)")

            self.messages.append(message)
            logger.info(f"📝 Assistant msg appended. History len: {len(self.messages)}")

            # ---- HANDLE TOOL CALLS ----
            for tool_call in message.tool_calls:
                full_name = tool_call.function.name
                args_raw = tool_call.function.arguments

                logger.info(f"🔧 Tool call: {full_name}")
                logger.info(f"📨 Raw args: {args_raw}")

                try:
                    args = json.loads(args_raw)
                    logger.info(f"📦 Parsed args: {json.dumps(args, indent=2)}")
                except Exception as e:
                    logger.error(f"❌ Failed parsing args: {e}")
                    args = {}

                if "_" not in full_name:
                    logger.error(f"❌ Invalid tool name: {full_name}")
                    continue

                service, tool = full_name.split("_", 1)
                logger.info(f"📍 Routing to: {service}.{tool}")

                if service not in self.sessions:
                    logger.error(f"❌ No session for {service}")
                    result_text = f"Error: Missing service {service}"
                else:
                    session = self.sessions[service]
                    logger.info("⏳ Calling remote tool...")
                    call_start = time.time()
                    try:
                        result = await session.call_tool(tool, args)
                        logger.info(f"⏱ Remote tool call took {time.time() - call_start:.4f}s")

                        if result.content and hasattr(result.content[0], 'text'):
                            result_text = result.content[0].text
                        else:
                            result_text = str(result)

                        logger.info("📤 TOOL RESULT:")
                        logger.info(result_text)

                    except Exception as e:
                        logger.error(f"❌ Tool call failed: {e}")
                        result_text = f"Tool error: {e}"

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text
                })
                logger.info(f"🧩 Tool result appended. History len: {len(self.messages)}")

            # ---- CALL OPENAI AGAIN ----
            logger.info("🧠 Sending tool results back to AI")

            ai2_start = time.time()
            response = self.openai.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                max_tokens=1000,
                messages=self.messages,
                tools=all_tools if all_tools else None
            )
            logger.info(f"⏱ OpenAI follow-up took {time.time() - ai2_start:.4f}s")

            message = response.choices[0].message

            logger.info("🧠 AI THOUGHT:")
            logger.info(str(message.content))

        # ------------------------------------------------------
        # END LOOP
        # ------------------------------------------------------
        final_text = message.content or ""
        logger.info("🏁 Loop complete. Final answer ready.")
        logger.info(f"⏱ Total query time: {time.time() - t0:.4f}s")
        logger.info(f"📦 Memory usage now: {memory_usage()}")

        self.messages.append({"role": "assistant", "content": final_text})

        return final_text

    # ------------------------------------------------------
    # CHAT LOOP
    # ------------------------------------------------------
    async def chat_loop(self):
        print("\n----------------------------------------------")
        print("🛰 MCP DEBUG CLIENT (ULTRA X-RAY) READY")
        print(f"🔗 Connected to: {', '.join(self.sessions.keys())}")
        print("----------------------------------------------")

        while True:
            try:
                query = await anyio.to_thread.run_sync(input, "\nQuery: ")
                query = query.strip()

                if query.lower() in ("quit", "exit"):
                    break

                if query.lower() == "clear":
                    self.messages = []
                    logger.info("🧹 History cleared")
                    continue

                ans = await self.process_query(query)
                print("\n🤖 FINAL ANSWER:\n" + ans)

            except EOFError:
                break
            except Exception as e:
                logger.critical(f"🔥 CRITICAL ERROR: {e}")


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
async def main():
    client = MultiMCPClient()

    async with AsyncExitStack() as stack:
        logger.info("🔌 Connecting to servers...")

        for name, url in KLAVIS_SERVERS.items():
            logger.info(f"➡️ Connecting to {name} at {url}")
            try:
                streams = await stack.enter_async_context(streamablehttp_client(url))
                session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))

                logger.info("⏳ Initializing session...")
                await session.initialize()

                client.sessions[name] = session
                logger.info(f"✅ Connected to {name}")
                logger.info(f"📦 Memory now: {memory_usage()}")

            except Exception as e:
                logger.error(f"❌ Failed to connect to {name}: {e}")

        if not client.sessions:
            logger.critical("❌ No servers connected. Exit.")
            return

        await client.chat_loop()


if __name__ == "__main__":
    try:
        anyio.run(main, backend="trio")
    except KeyboardInterrupt:
        pass
