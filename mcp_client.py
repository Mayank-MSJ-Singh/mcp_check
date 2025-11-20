"""
Multi-Server MCP Client (NUCLEAR X-RAY EDITION)
- EVERYTHING logged: lifecycle, data, timing, memory, tokens, per-variable dumps
- Uses tiktoken for deterministic token counts
- Defensive: won't crash if some SDK fields are missing
- Use sparingly (very verbose logs)

Primary purpose:
- Find today's events (via gcalendar tools) and send a notification to
  mayank.msj.singh@gmail.com (the logic can be implemented by the model/tool calls).
"""

import json
import logging
import time
import tracemalloc
import anyio
import tiktoken
import traceback
from contextlib import AsyncExitStack
from typing import Dict, Any

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
    "slack": "https://slack-mcp-server.klavis.ai/mcp/?instance_id=bfb2bca1-e73b-4c9d-9338-de1ecd35f4ea",
    "attio": "https://attio-mcp-server.klavis.ai/mcp/?instance_id=82c94b49-2981-4ab2-9433-1f081a36f22c",
    "hackerNews": "https://hacker-news-mcp-server.klavis.ai/mcp/?instance_id=000c2de5-7296-4417-97e2-6082e77a0050",
}

load_dotenv()

# -------------------------------
# LOGGING SETUP (NUCLEAR)
# -------------------------------
# This is intentionally very verbose (DEBUG).
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)

# Reduce third-party noise where possible
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("NUCLEAR-XRAY-MCP")

tracemalloc.start()

# Try to import psutil for richer system stats; optional
try:
    import psutil
    PSUTIL_OK = True
except Exception:
    PSUTIL_OK = False


def pretty_json(obj: Any, max_len: int = 2000) -> str:
    try:
        s = json.dumps(obj, indent=2, default=str)
        if len(s) > max_len:
            return s[:max_len] + "...(truncated)"
        return s
    except Exception as e:
        return f"<unserializable: {e}>"


def memory_usage() -> str:
    try:
        current, peak = tracemalloc.get_traced_memory()
        mem = f"{current/1024/1024:.3f}MB (peak {peak/1024/1024:.3f}MB)"
    except Exception:
        mem = "N/A"
    if PSUTIL_OK:
        try:
            p = psutil.Process()
            rss = p.memory_info().rss / 1024 / 1024
            cpu = psutil.cpu_percent(interval=0.0)
            return f"{mem} | RSS {rss:.3f}MB | CPU% {cpu:.1f}"
        except Exception:
            return mem
    return mem


# =====================================================
#                 CLIENT CLASS
# =====================================================
class MultiMCPClient:
    def __init__(self):
        logger.info("🧩 INIT MultiMCPClient")
        self.sessions: Dict[str, ClientSession] = {}
        self.openai = OpenAI()
        self.messages = []
        # tiktoken setup
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4.1-mini-2025-04-14")
            logger.debug("tiktoken encoding initialized for model gpt-4.1-mini-2025-04-14")
        except Exception as e:
            # fallback to a generic encoding if model-specific not found
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
                logger.debug("tiktoken fallback encoding initialized: cl100k_base")
            except Exception as e2:
                logger.error(f"tiktoken failed to initialize: {e} / {e2}")
                self.encoding = None
        self.total_tokens_used = 0
        self.query_counter = 0
        logger.debug(f"memory on init: {memory_usage()}")

    # -----------------------------
    # Low-level helpers
    # -----------------------------
    def _count_tokens_from_text(self, text: str) -> int:
        if not self.encoding:
            return 0
        try:
            enc = self.encoding.encode(text)
            logger.debug(f"encoded text len {len(enc)} tokens, bytes {len(text.encode('utf-8'))}")
            return len(enc)
        except Exception as e:
            logger.error(f"tiktoken encode error: {e}")
            return 0

    def count_tokens(self, messages) -> int:
        """
        Count tokens for a list of messages. Uses a simple concatenation approach
        that replicates a reasonable token estimate (role + content). Extremely verbose.
        """
        try:
            pieces = []
            for m in messages:
                if isinstance(m, dict):
                    role = str(m.get("role", ""))
                    content = str(m.get("content", ""))
                    # Also include tool_call_id if present (keeps tokens consistent)
                    tcid = m.get("tool_call_id", "")
                    pieces.append(f"{role}:{tcid}:{content}\n")
                else:
                    pieces.append(str(m) + "\n")
            big = "".join(pieces)
            token_count = self._count_tokens_from_text(big)
            logger.debug(f"count_tokens: messages_count={len(messages)} chars={len(big)} tokens={token_count}")
            return token_count
        except Exception as e:
            logger.error(f"count_tokens failed: {e}")
            return 0

    def log_response_usage_if_any(self, label: str, response):
        """
        Safely try to extract response.usage (prompt/completion/total).
        Many SDKs put this in different spots; be defensive.
        """
        try:
            usage = getattr(response, "usage", None)
            if not usage:
                # maybe response.to_dict() exists
                d = {}
                try:
                    d = response.to_dict()
                except Exception:
                    d = {}
                usage = d.get("usage") or d.get("meta", {}).get("usage")
            if usage:
                try:
                    pt = usage.get("prompt_tokens", getattr(usage, "prompt_tokens", None))
                    ct = usage.get("completion_tokens", getattr(usage, "completion_tokens", None))
                    tt = usage.get("total_tokens", getattr(usage, "total_tokens", None))
                except Exception:
                    pt = getattr(usage, "prompt_tokens", None)
                    ct = getattr(usage, "completion_tokens", None)
                    tt = getattr(usage, "total_tokens", None)
                logger.info(f"📊 TOKEN USAGE ({label}): prompt={pt} completion={ct} total={tt}")
            else:
                logger.debug(f"📊 TOKEN USAGE ({label}): not present in response")
        except Exception as e:
            logger.error(f"Error reading response usage ({label}): {e}")

    # -----------------------------
    # Ultra logging helpers
    # -----------------------------
    def dump_state(self):
        """
        Giant snapshot of relevant internal state for debugging.
        """
        try:
            info = {
                "sessions": list(self.sessions.keys()),
                "messages_count": len(self.messages),
                "last_10_messages": self.messages[-10:],
                "total_tokens_used": self.total_tokens_used,
                "memory": memory_usage(),
            }
            logger.debug("STATE SNAPSHOT:\n" + pretty_json(info, max_len=5000))
        except Exception as e:
            logger.error(f"dump_state failed: {e}")

    def safe_json(self, obj, label="obj", max_len=2000):
        try:
            s = pretty_json(obj, max_len=max_len)
            logger.debug(f"{label}: {s}")
        except Exception as e:
            logger.debug(f"{label}: <unserializable: {e}>")

    # =====================================================
    #               PROCESS A QUERY (NUCLEAR)
    # =====================================================
    async def process_query(self, query: str) -> str:
        """
        Main logic with absolutely maximal logging.
        """
        self.query_counter += 1
        qid = self.query_counter
        logger.info(f"\n\n==================== NEW QUERY #{qid} ====================")
        logger.info(f"User query: {query}")
        logger.debug(f"Full env/debug memory: {memory_usage()}")
        start_time = time.time()

        # Append user message
        user_msg = {"role": "user", "content": query}
        self.messages.append(user_msg)
        logger.debug(f"Appended user message to history (len={len(self.messages)})")
        self.safe_json(user_msg, label="user_msg", max_len=800)

        # Token snapshot pre-call
        tokens_before = self.count_tokens(self.messages)
        logger.info(f"🔢 TOKENS BEFORE ANY OPENAI CALL: {tokens_before}")

        # --- AGGREGATE TOOLS ---
        logger.info("🔍 Aggregating tools from all connected servers (detailed)...")
        agg_start = time.time()
        all_tools = []
        per_server_tool_counts = {}

        for service_name, session in self.sessions.items():
            server_start = time.time()
            logger.debug(f"➡️ Fetching tools from server '{service_name}' (session obj: {repr(session)})")
            try:
                response = await session.list_tools()
                tools = getattr(response, "tools", []) or []
                per_server_tool_counts[service_name] = len(tools)
                logger.info(f"  {service_name}: {len(tools)} tool(s) returned")
                # Deep log each tool
                for t_idx, tool in enumerate(tools):
                    try:
                        tool_info = {
                            "index": t_idx,
                            "name": getattr(tool, "name", None),
                            "description": getattr(tool, "description", None),
                            "inputSchema": getattr(tool, "inputSchema", None),
                            "raw": tool  # may be unserializable
                        }
                        logger.debug(f"    TOOL: {tool_info['name']} -> {pretty_json(tool_info, max_len=2000)}")
                        prefixed_name = f"{service_name}_{tool.name}"
                        all_tools.append({
                            "type": "function",
                            "function": {
                                "name": prefixed_name,
                                "description": f"[{service_name.upper()}] {tool.description}",
                                "parameters": tool.inputSchema
                            }
                        })
                    except Exception as e:
                        logger.error(f"    ❌ Error logging tool from {service_name}: {e}")
                        logger.debug(traceback.format_exc())
            except Exception as e:
                logger.error(f"❌ Error listing tools from {service_name}: {e}")
                logger.debug(traceback.format_exc())
            logger.debug(f"  Fetch tools took {time.time() - server_start:.4f}s for {service_name}")

        logger.info(f"📦 TOTAL TOOLS AGGREGATED: {len(all_tools)}")
        logger.debug(f"Per-server tool counts: {pretty_json(per_server_tool_counts)}")
        logger.info(f"⏱ Tool aggregation time: {time.time() - agg_start:.4f}s")
        self.dump_state()

        # Dump the full tool payload size for debugging
        try:
            tools_serialized = json.dumps(all_tools)
            logger.debug(f"All tools serialized size: {len(tools_serialized)} bytes | preview: {tools_serialized[:2000]}...")
        except Exception as e:
            logger.debug(f"Could not serialize all_tools: {e}")

        # ------------------------------------------------------
        # 1) FIRST OPENAI CALL (with full logs)
        # ------------------------------------------------------
        logger.info("🧠 Sending initial request to OpenAI (full payload logging)")
        try:
            # Log last N messages fully (huge)
            try:
                logger.debug("MESSAGES PAYLOAD (last 50):\n" + pretty_json(self.messages[-50:], max_len=8000))
            except Exception as e:
                logger.debug(f"Failed to serialize messages for debug: {e}")

            call_start = time.time()
            response = self.openai.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                max_tokens=1000,
                messages=self.messages,
                tools=all_tools or None
            )
            call_time = time.time() - call_start
            logger.info(f"✅ OpenAI initial call completed in {call_time:.4f}s")
            # Raw response dump (may be large)
            try:
                rd = response.to_dict()
                logger.debug("RAW OPENAI RESPONSE (dict preview):\n" + pretty_json(rd, max_len=10000))
            except Exception:
                try:
                    logger.debug("RAW OPENAI RESPONSE (repr):\n" + repr(response)[:4000])
                except Exception:
                    logger.debug("RAW OPENAI RESPONSE: <unserializable>")

            # Try to log usage if present
            self.log_response_usage_if_any("initial_call", response)

            # Defensive extraction for message & tool_calls
            try:
                message = response.choices[0].message
            except Exception:
                # fallback shapes
                try:
                    choices = getattr(response, "choices", [])
                    message = choices[0]["message"] if isinstance(choices, (list, tuple)) and choices else None
                except Exception:
                    message = None

            if message is None:
                logger.error("❌ No assistant message found in response. Aborting query processing.")
                logger.debug(f"Full response object: {repr(response)[:2000]}")
                return "Error: Empty assistant response."

            # Token counting for assistant reply
            assistant_content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else "")
            reply_tokens = self.count_tokens([{"role": "assistant", "content": assistant_content}])
            self.total_tokens_used += reply_tokens
            logger.info(f"🔢 FIRST ASSISTANT REPLY TOKENS: {reply_tokens}")
            logger.info(f"🔢 TOTAL TOKENS USED (running): {self.total_tokens_used}")

            # Log tool calls (if any)
            tcs = getattr(message, "tool_calls", None)
            if tcs is None and isinstance(message, dict):
                tcs = message.get("tool_calls")
            tcs_len = len(tcs) if tcs else 0
            logger.info(f"🔧 Assistant requested tool calls: {tcs_len}")
            if tcs_len:
                logger.debug("Tool calls payload:\n" + pretty_json(tcs, max_len=8000))

        except Exception as e:
            logger.error(f"🔥 OpenAI initial call failed: {e}")
            logger.debug(traceback.format_exc())
            return f"OpenAI call failed: {e}"

        # ------------------------------------------------------
        # 2) AGENTIC LOOP - FULLY LOGGED (Nuclear)
        # ------------------------------------------------------
        loop_index = 0
        last_loop_end = time.time()
        while True:
            # retrieve tool_calls defensively
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls is None and isinstance(message, dict):
                tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                logger.info("No tool calls returned by assistant. Breaking agentic loop.")
                break

            loop_index += 1
            loop_start = time.time()
            logger.info(f"\n--- AGENTIC LOOP #{loop_index} START (tool_calls={len(tool_calls)}) ---")
            logger.debug(f"Time since last loop end: {loop_start - last_loop_end:.6f}s")
            last_loop_end = loop_start

            # append the assistant message to the history (so model sees it)
            try:
                self.messages.append(message)
                logger.debug(f"Appended assistant message to history (len={len(self.messages)})")
                self.safe_json({"assistant_message_preview": getattr(message, "content", "<no content>")}, label="assistant_preview", max_len=2000)
            except Exception as e:
                logger.error(f"Failed to append assistant message: {e}")

            # process each tool call
            for tc_idx, tool_call in enumerate(tool_calls):
                logger.info(f"→ Processing tool call #{tc_idx+1}/{len(tool_calls)}")
                try:
                    # deep log tool_call object
                    try:
                        logger.debug("tool_call object: " + pretty_json(tool_call, max_len=5000))
                    except Exception:
                        logger.debug(f"tool_call repr: {repr(tool_call)[:2000]}")

                    # retrieve function name and arguments safely
                    func = getattr(tool_call, "function", None) or (tool_call.get("function") if isinstance(tool_call, dict) else None)
                    if not func:
                        logger.error("tool_call.function missing. Skipping this tool_call.")
                        continue

                    full_name = getattr(func, "name", None) or (func.get("name") if isinstance(func, dict) else None)
                    raw_args = getattr(func, "arguments", None) or (func.get("arguments") if isinstance(func, dict) else None)
                    tool_call_id = getattr(tool_call, "id", None) or (tool_call.get("id") if isinstance(tool_call, dict) else None)

                    logger.debug(f"tool_call_id: {tool_call_id} full_name: {full_name} raw_args_preview: {str(raw_args)[:2000]}")

                    # parse args
                    parsed_args = {}
                    if raw_args:
                        try:
                            parsed_args = json.loads(raw_args)
                            logger.debug("Parsed tool args: " + pretty_json(parsed_args, max_len=4000))
                        except Exception as e:
                            logger.warning(f"Could not parse tool args as JSON: {e}")
                            parsed_args = {"_raw": raw_args}

                    # validate name format: service_tool
                    if not full_name or "_" not in full_name:
                        logger.error(f"Malformed tool name: {full_name} (expected service_tool). Skipping.")
                        tool_result = f"Error: malformed tool name: {full_name}"
                    else:
                        service, actual_tool_name = full_name.split("_", 1)
                        logger.info(f"Routing tool call to service='{service}', tool='{actual_tool_name}'")
                        # ensure session exists
                        if service not in self.sessions:
                            logger.error(f"Service '{service}' not connected.")
                            tool_result = f"Error: service {service} not connected"
                        else:
                            session = self.sessions[service]
                            # before-call state
                            pre_call_tokens = self.count_tokens(self.messages[-50:])  # last 50 messages context
                            logger.debug(f"Tokens in last-50-messages before tool exec: {pre_call_tokens}")
                            call_time_start = time.time()
                            try:
                                logger.info(f"⏳ Executing remote tool {service}.{actual_tool_name} (id={tool_call_id})")
                                tool_exec_result = await session.call_tool(actual_tool_name, parsed_args)
                                call_time = time.time() - call_time_start
                                logger.info(f"✅ Remote tool call completed in {call_time:.4f}s")
                                # deep inspect result
                                try:
                                    # result.content sometimes is list of objects with .text
                                    if hasattr(tool_exec_result, "content"):
                                        content = tool_exec_result.content
                                    else:
                                        # fallback
                                        content = getattr(tool_exec_result, "result", None) or str(tool_exec_result)
                                    logger.debug("Raw tool exec result preview: " + pretty_json(content, max_len=8000))
                                except Exception:
                                    logger.debug("Tool exec result repr: " + repr(tool_exec_result)[:2000])

                                # extract human-readable text if possible
                                result_text = ""
                                try:
                                    if hasattr(tool_exec_result, "content") and isinstance(tool_exec_result.content, (list, tuple)) and len(tool_exec_result.content) > 0:
                                        first = tool_exec_result.content[0]
                                        # common field name is 'text'
                                        result_text = getattr(first, "text", None) or (first.get("text") if isinstance(first, dict) else None)
                                        if result_text is None:
                                            result_text = pretty_json(first, max_len=2000)
                                    else:
                                        result_text = str(tool_exec_result)
                                except Exception:
                                    result_text = str(tool_exec_result)

                                logger.info(f"📤 Tool result (trimmed): {result_text[:1500]}")

                            except Exception as e:
                                call_time = time.time() - call_time_start
                                logger.error(f"❌ Remote tool call failed after {call_time:.4f}s: {e}")
                                logger.debug(traceback.format_exc())
                                result_text = f"Tool execution error: {e}"

                            tool_result = result_text

                    # append tool role message
                    try:
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": tool_result
                        }
                        self.messages.append(tool_msg)
                        logger.debug(f"Appended tool result to messages (len={len(self.messages)})")
                        self.safe_json(tool_msg, label="tool_msg", max_len=2000)
                    except Exception as e:
                        logger.error(f"Failed to append tool result message: {e}")

                except Exception as e:
                    logger.error(f"Unexpected error processing tool_call: {e}")
                    logger.debug(traceback.format_exc())

            # After processing all tool calls, call OpenAI again with the augmented history
            try:
                logger.info("🧠 Sending tool results back to OpenAI for next decision step")
                # log a compact preview of the last several messages that will be sent
                try:
                    preview_msgs = self.messages[-60:]
                    logger.debug("NEXT OPENAI CALL MESSAGES PREVIEW:\n" + pretty_json(preview_msgs, max_len=12000))
                except Exception as e:
                    logger.debug(f"Could not serialize preview messages: {e}")

                ai_loop_start = time.time()
                response = self.openai.chat.completions.create(
                    model="gpt-4.1-mini-2025-04-14",
                    max_tokens=1000,
                    messages=self.messages,
                    tools=all_tools or None
                )
                ai_loop_time = time.time() - ai_loop_start
                logger.info(f"✅ OpenAI loop call completed in {ai_loop_time:.4f}s")
                # dump raw response defensively
                try:
                    logger.debug("RAW LOOP RESPONSE (dict preview):\n" + pretty_json(response.to_dict(), max_len=12000))
                except Exception:
                    logger.debug("RAW LOOP RESPONSE (repr):\n" + repr(response)[:4000])

                self.log_response_usage_if_any(f"loop_call_{loop_index}", response)

                # extract message
                try:
                    message = response.choices[0].message
                except Exception:
                    try:
                        choices = getattr(response, "choices", [])
                        message = choices[0]["message"] if isinstance(choices, (list, tuple)) and choices else None
                    except Exception:
                        message = None

                if message is None:
                    logger.error("❌ No assistant message in loop response. Breaking.")
                    break

                # count tokens for assistant's new reply
                assistant_content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else "")
                loop_reply_tokens = self.count_tokens([{"role": "assistant", "content": assistant_content}])
                self.total_tokens_used += loop_reply_tokens
                logger.info(f"🔢 LOOP #{loop_index} assistant tokens: {loop_reply_tokens}")
                logger.info(f"🔢 TOTAL TOKENS USED (running): {self.total_tokens_used}")

                # loop delta timings
                loop_end = time.time()
                logger.debug(f"Loop #{loop_index} duration: {loop_end - loop_start:.4f}s | total so far: {loop_end - start_time:.4f}s")

            except Exception as e:
                logger.error(f"🔥 OpenAI loop call failed: {e}")
                logger.debug(traceback.format_exc())
                break

        # END agentic loop
        logger.info("\n--- AGENTIC LOOP COMPLETE ---")
        total_time = time.time() - start_time
        logger.info(f"🏁 Query #{qid} complete in {total_time:.4f}s")
        logger.info(f"📦 Memory at end: {memory_usage()}")
        logger.info(f"🧮 Total tokens used (running sum from tiktoken calcs): {self.total_tokens_used}")

        # count final tokens in entire conversation (safe snapshot)
        try:
            final_conv_tokens = self.count_tokens(self.messages)
            logger.info(f"🧾 Final conversation token count (full history): {final_conv_tokens}")
        except Exception as e:
            logger.error(f"Failed to count final conversation tokens: {e}")

        # final assistant content
        try:
            final_text = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else "")
            if final_text:
                logger.debug("FINAL ASSISTANT CONTENT PREVIEW:\n" + (final_text[:4000] if isinstance(final_text, str) else pretty_json(final_text, max_len=4000)))
                # append final assistant content in history as assistant role (if not already)
                self.messages.append({"role": "assistant", "content": final_text})
            else:
                final_text = "Task completed (No final text response)."
                logger.info("Assistant returned no final 'content' field.")
        except Exception as e:
            logger.error(f"Error retrieving final assistant content: {e}")
            final_text = f"Error retrieving final assistant content: {e}"

        logger.info(f"==================== END QUERY #{qid} ====================\n\n")
        return final_text

    # =====================================================
    #                    CHAT LOOP
    # =====================================================
    async def chat_loop(self):
        print("\n----------------------------------------------")
        print("🛰 NUCLEAR MCP CLIENT READY")
        print(f"🔗 Connected to: {', '.join(self.sessions.keys()) if self.sessions else '<none>'}")
        print("----------------------------------------------")

        while True:
            try:
                query = await anyio.to_thread.run_sync(input, "\nQuery: ")
                query = query.strip()

                if query.lower() in ("quit", "exit"):
                    logger.info("Received exit command. Shutting down chat loop.")
                    break

                if query.lower() == "clear":
                    self.messages = []
                    self.total_tokens_used = 0
                    logger.info("🧹 History and token counters cleared.")
                    continue

                # A quick safety: dump current sessions
                logger.debug(f"Current sessions: {list(self.sessions.keys())}")
                ans = await self.process_query(query)
                print("\n🤖 FINAL ANSWER:\n")
                print(ans)
            except EOFError:
                logger.info("EOF received, exiting.")
                break
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received, exiting.")
                break
            except Exception as e:
                logger.critical(f"🔥 CRITICAL ERROR (chat_loop): {e}")
                logger.debug(traceback.format_exc())
                # don't break; allow continued use
                continue


# =====================================================
#                     MAIN
# =====================================================
async def main():
    client = MultiMCPClient()

    async with AsyncExitStack() as stack:
        logger.info("🔌 Attempting to connect to configured MCP servers...")

        for name, url in KLAVIS_SERVERS.items():
            logger.info(f"➡️ Connecting to '{name}' at {url}")
            try:
                connect_start = time.time()
                streams = await stack.enter_async_context(streamablehttp_client(url))
                session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))

                logger.debug(f"Session object for {name}: {repr(session)[:400]}")
                logger.info(f"⏳ Initializing session for {name}...")
                await session.initialize()
                client.sessions[name] = session
                ctime = time.time() - connect_start
                logger.info(f"✅ Connected & initialized '{name}' in {ctime:.4f}s")
                logger.debug(f"Memory after connect {name}: {memory_usage()}")
            except Exception as e:
                logger.error(f"❌ Failed connecting/initializing {name}: {e}")
                logger.debug(traceback.format_exc())

        if not client.sessions:
            logger.critical("❌ No MCP sessions were established. Exiting.")
            return

        # Optionally, perform an automated query on startup to find today's events and send email
        # (The actual "send email" should be performed via the remote gmail tool through MCP.)
        logger.info("Client ready. Enter queries at the prompt or let the model orchestrate tasks.")
        await client.chat_loop()


if __name__ == "__main__":
    try:
        anyio.run(main, backend="trio")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt main - exiting gracefully.")
    except Exception as e:
        logger.critical(f"Uncaught exception in main: {e}")
        logger.debug(traceback.format_exc())
