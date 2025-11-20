"""
Multi-Server MCP Client (C-MODE FIXED - SAFE TRUNCATION)
- Plain English logs
- Real token counting (messages + tools)
- Tool results sanitized to plain text (no raw objects)
- Retry-on-400 with sanitized/truncated payload (preserves tool_call_id, type & tool_calls)
- Safe shutdown wrapper
"""

import json
import logging
import time
import anyio
import tiktoken
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Dict, Any, List

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import OpenAI

# ---------------------------
# Servers
# ---------------------------
KLAVIS_SERVERS = {
    "youtube": "https://youtube-mcp-server.klavis.ai/mcp/?instance_id=e5f0b026-f5db-402e-a6b6-62000d3444ab"
}

load_dotenv()

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("CLEAN-MCP")

# ---------------------------
# Safe cleanup wrapper
# ---------------------------
@asynccontextmanager
async def safe_stack():
    try:
        yield
    except Exception:
        # suppress shutdown-related exceptions (task-group race conditions)
        pass

# ---------------------------
# Helpers
# ---------------------------
def safe_json_dump(obj: Any, max_len: int = 10000) -> str:
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
        if len(s) > max_len:
            return s[:max_len] + "...(truncated)"
        return s
    except Exception:
        try:
            s = repr(obj)
            return s[:max_len] + ("...(truncated)" if len(s) > max_len else "")
        except Exception:
            return "<unserializable>"

def sanitize_tool_result(result_obj: Any, max_chars: int = 8000) -> str:
    """
    Convert whatever the tool returned into a readable, plain text string.
    Truncate if very large. Use json.dumps with default=str for structured objects.
    """
    if result_obj is None:
        return ""
    # If object has .content with list of objects, try to extract text fields
    try:
        if hasattr(result_obj, "content") and isinstance(result_obj.content, (list, tuple)):
            parts = []
            for item in result_obj.content:
                # prefer .text attribute or dictionary 'text' key
                text = None
                if hasattr(item, "text"):
                    text = getattr(item, "text")
                elif isinstance(item, dict) and "text" in item:
                    text = item.get("text")
                else:
                    text = safe_json_dump(item, max_len=1000)
                if text is None:
                    text = ""
                parts.append(str(text))
            joined = "\n".join(parts)
            return joined[:max_chars] + ("...(truncated)" if len(joined) > max_chars else joined)
    except Exception:
        pass

    # Fallback: json dump or str()
    try:
        s = json.dumps(result_obj, default=str, ensure_ascii=False)
    except Exception:
        s = str(result_obj)
    if len(s) > max_chars:
        return s[:max_chars] + "...(truncated)"
    return s

# ---------------------------
# Client
# ---------------------------
class MultiMCPClient:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.openai = OpenAI()
        self.messages: List[Dict[str, Any]] = []
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4.1-mini-2025-04-14")
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # Count tokens in the exact payload we'll send to OpenAI
    def count_payload_tokens(self, messages, tools) -> int:
        payload = {
            "model": "gpt-4.1-mini-2025-04-14",
            "messages": messages,
            "tools": tools
        }
        text = json.dumps(payload, default=str, ensure_ascii=False)
        return len(self.encoding.encode(text))

    def count_output_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text or ""))

    async def process_query(self, query: str):
        log.info("--------------------------------------------------")
        log.info(f"New query: '{query}'")
        # Append user message
        self.messages.append({"role": "user", "content": query})

        # Build tools list (serializable)
        tools = []
        for name, session in self.sessions.items():
            try:
                resp = await session.list_tools()
                tools_found = getattr(resp, "tools", []) or []
                for t in tools_found:
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": f"{name}_{t.name}",
                            "description": t.description,
                            "parameters": t.inputSchema
                        }
                    })
            except Exception as e:
                log.warning(f"Could not list tools from {name}: {e}")

        log.info(f"Tools available to AI: {len(tools)}")

        # Helper to build a sanitized copy of a single message preserving required fields
        def build_sanitized_message(m: Dict[str, Any], content_limit: int = 2000) -> Dict[str, Any]:
            sanitized = {}
            # Copy role always
            sanitized["role"] = m.get("role", "")
            # Preserve tool_call_id if present
            if "tool_call_id" in m:
                sanitized["tool_call_id"] = m["tool_call_id"]
            # Copy and truncate content
            content = str(m.get("content", "") or "")
            if len(content) > content_limit:
                content = content[:content_limit] + "...(truncated)"
            sanitized["content"] = content
            # If the original assistant message had tool_calls, preserve structure but truncate arguments inside
            if "tool_calls" in m and isinstance(m["tool_calls"], (list, tuple)):
                truncated_calls = []
                for tc in m["tool_calls"]:
                    try:
                        # keep id and function.name; truncate arguments string if present
                        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                        func = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
                        func_name = None
                        args_raw = None
                        if isinstance(func, dict):
                            func_name = func.get("name")
                            args_raw = func.get("arguments")
                        else:
                            func_name = getattr(func, "name", None)
                            args_raw = getattr(func, "arguments", None)
                        # truncate arguments string to avoid huge payloads
                        args_str = str(args_raw) if args_raw is not None else ""
                        if len(args_str) > 200:
                            args_str = args_str[:200] + "...(truncated)"
                        truncated_calls.append({
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": func_name, "arguments": args_str}
                        })
                    except Exception:
                        # fallback minimal
                        truncated_calls.append({"id": None, "type": "function", "function": {"name": None, "arguments": ""}})
                sanitized["tool_calls"] = truncated_calls
            return sanitized

        # Function to call OpenAI safely and retry-on-400 with sanitized/truncated payload
        def openai_call_with_retry(messages_payload, tools_payload, attempt_label="first"):
            """
            Returns (response, sent_tokens_estimate) or raises.
            The sanitized retry preserves required keys: role, content, tool_call_id, tool_calls.
            """
            sent_tokens = self.count_payload_tokens(messages_payload, tools_payload)
            log.info(f"{attempt_label.capitalize()} call → approx {sent_tokens} tokens (payload)")
            try:
                resp = self.openai.chat.completions.create(
                    model="gpt-4.1-mini-2025-04-14",
                    messages=messages_payload,
                    tools=tools_payload
                )
                return resp, sent_tokens
            except Exception as e:
                # Try one retry with sanitized/truncated messages (but preserve tool_call_id & tool_calls)
                err_str = str(e)
                log.warning(f"OpenAI call failed ({attempt_label}): {err_str}. Attempting sanitized retry.")
                sanitized_messages = []
                for m in messages_payload:
                    if isinstance(m, dict):
                        sanitized_messages.append(build_sanitized_message(m, content_limit=2000))
                    else:
                        # fallback
                        sanitized_messages.append({"role": "user", "content": str(m)[:2000]})
                try:
                    sent_tokens2 = self.count_payload_tokens(sanitized_messages, tools_payload)
                    log.info(f"Sanitized retry → approx {sent_tokens2} tokens")
                    resp2 = self.openai.chat.completions.create(
                        model="gpt-4.1-mini-2025-04-14",
                        messages=sanitized_messages,
                        tools=tools_payload
                    )
                    return resp2, sent_tokens2
                except Exception as e2:
                    log.error(f"Sanitized retry failed: {e2}")
                    raise

        # ---------- initial call ----------
        try:
            resp, sent_tokens = openai_call_with_retry(self.messages, tools, attempt_label="initial")
        except Exception as e:
            log.error(f"Initial OpenAI call failed: {e}")
            return "OpenAI call failed."

        # Extract assistant message and token accounting
        try:
            assistant_msg = resp.choices[0].message
        except Exception:
            # defensive extracting for other shapes
            choices = getattr(resp, "choices", []) or []
            assistant_msg = choices[0].get("message") if choices else {"content": ""}

        assistant_content = getattr(assistant_msg, "content", None) or (assistant_msg.get("content") if isinstance(assistant_msg, dict) else "")
        out_tokens = self.count_output_tokens(assistant_content)
        log.info(f"AI replied using ~{out_tokens} tokens (assistant output)")

        self.total_input_tokens += sent_tokens
        self.total_output_tokens += out_tokens

        # Tools requested?
        tool_calls = getattr(assistant_msg, "tool_calls", None)
        if tool_calls is None and isinstance(assistant_msg, dict):
            tool_calls = assistant_msg.get("tool_calls", [])
        tool_calls = tool_calls or []
        log.info(f"AI requested {len(tool_calls)} tool calls")

        # Append assistant message to history — preserve tool_calls as returned by model
        # Keep the assistant object intact (role, content, tool_calls if present)
        assistant_history_entry: Dict[str, Any] = {"role": "assistant", "content": assistant_content}
        if tool_calls:
            # normalize tool_calls to serializable dicts if possible
            safe_calls = []
            for tc in tool_calls:
                try:
                    # if tc is an object or dict, pull id and function details
                    if isinstance(tc, dict):
                        # ensure "type" present
                        tc_copy = dict(tc)
                        if "type" not in tc_copy:
                            tc_copy["type"] = "function"
                        safe_calls.append(tc_copy)
                    else:
                        # try to make a dict
                        func = getattr(tc, "function", None)
                        fid = getattr(tc, "id", None)
                        func_name = getattr(func, "name", None) if func is not None else None
                        args = getattr(func, "arguments", None) if func is not None else None
                        safe_calls.append({
                            "id": fid,
                            "type": "function",
                            "function": {"name": func_name, "arguments": args}
                        })
                except Exception:
                    safe_calls.append({"id": None, "type": "function", "function": {"name": None, "arguments": None}})
            assistant_history_entry["tool_calls"] = safe_calls
        self.messages.append(assistant_history_entry)

        # ---------- Agentic loop (support multiple rounds) ----------
        round_idx = 0
        while tool_calls:
            round_idx += 1
            log.info(f"--- Agentic loop round {round_idx} (processing {len(tool_calls)} calls) ---")
            for tc in tool_calls:
                # Extract name/args defensively
                func = getattr(tc, "function", None) if not isinstance(tc, dict) else tc.get("function")
                if func is None:
                    log.warning("Malformed tool call (no function). Skipping.")
                    continue
                full_name = getattr(func, "name", None) if not isinstance(func, dict) else func.get("name")
                raw_args = getattr(func, "arguments", None) if not isinstance(func, dict) else func.get("arguments")
                tool_call_id = getattr(tc, "id", None) if not isinstance(tc, dict) else tc.get("id")

                if not full_name or "_" not in full_name:
                    log.warning(f"Malformed tool name: {full_name}. Skipping.")
                    continue

                service, tool_name = full_name.split("_", 1)
                log.info(f"Calling tool: {service}.{tool_name}")

                session = self.sessions.get(service)
                if not session:
                    log.warning(f"Service {service} not connected. Appending error to history.")
                    self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": f"Error: service {service} not connected"})
                    continue

                # parse args safely
                args = {}
                if raw_args:
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = {"_raw": str(raw_args)}

                # call remote tool
                try:
                    result = await session.call_tool(tool_name, args)
                    sanitized = sanitize_tool_result(result, max_chars=8000)
                    log.info(f"Tool result length: {len(sanitized)} chars")
                except Exception as e:
                    sanitized = f"Tool execution error: {e}"
                    log.error(sanitized)

                # append sanitized result — keep tool_call_id intact
                tool_history_entry = {"role": "tool", "tool_call_id": tool_call_id, "content": sanitized}
                self.messages.append(tool_history_entry)

            # After processing all tool calls, call OpenAI again
            try:
                resp2, sent_tokens2 = openai_call_with_retry(self.messages, tools, attempt_label=f"loop_{round_idx}")
            except Exception as e:
                log.error(f"OpenAI follow-up failed after tools: {e}")
                break

            # extract assistant msg
            try:
                assistant_msg = resp2.choices[0].message
            except Exception:
                choices = getattr(resp2, "choices", []) or []
                assistant_msg = choices[0].get("message") if choices else {"content": ""}

            assistant_content = getattr(assistant_msg, "content", None) or (assistant_msg.get("content") if isinstance(assistant_msg, dict) else "")
            out_tokens2 = self.count_output_tokens(assistant_content)
            log.info(f"AI follow-up replied using ~{out_tokens2} tokens")

            # account tokens
            self.total_input_tokens += sent_tokens2
            self.total_output_tokens += out_tokens2

            # append assistant message (preserve tool_calls if present)
            tool_calls = getattr(assistant_msg, "tool_calls", None)
            if tool_calls is None and isinstance(assistant_msg, dict):
                tool_calls = assistant_msg.get("tool_calls", [])
            tool_calls = tool_calls or []

            assistant_history_entry = {"role": "assistant", "content": assistant_content}
            if tool_calls:
                # safe copy as above
                safe_calls = []
                for tc in tool_calls:
                    try:
                        if isinstance(tc, dict):
                            tc_copy = dict(tc)
                            if "type" not in tc_copy:
                                tc_copy["type"] = "function"
                            safe_calls.append(tc_copy)
                        else:
                            func = getattr(tc, "function", None)
                            fid = getattr(tc, "id", None)
                            func_name = getattr(func, "name", None) if func is not None else None
                            args = getattr(func, "arguments", None) if func is not None else None
                            safe_calls.append({
                                "id": fid,
                                "type": "function",
                                "function": {"name": func_name, "arguments": args}
                            })
                    except Exception:
                        safe_calls.append({"id": None, "type": "function", "function": {"name": None, "arguments": None}})
                assistant_history_entry["tool_calls"] = safe_calls
            self.messages.append(assistant_history_entry)

            log.info(f"AI requested {len(tool_calls)} tool calls in this follow-up")

            # next iteration will process these tool_calls
            # if none, loop exits naturally

        # ---------- done ----------
        log.info("Agentic loop complete.")
        log.info(f"Input tokens total (approx): {self.total_input_tokens}")
        log.info(f"Output tokens total (approx): {self.total_output_tokens}")
        log.info(f"Combined tokens (approx): {self.total_input_tokens + self.total_output_tokens}")

        # Return last assistant content (or generic)
        final_text = self.messages[-1].get("content") if self.messages else "(no response)"
        return final_text

    async def chat_loop(self):
        print("MCP Client Ready.\n")
        while True:
            q = await anyio.to_thread.run_sync(input, "\nQuery: ")
            q = q.strip()
            if q.lower() in ("quit", "exit"):
                break
            out = await self.process_query(q)
            print("\nAssistant:", out)

# ---------------------------
# Main
# ---------------------------
async def main():
    client = MultiMCPClient()
    async with safe_stack(), AsyncExitStack() as stack:
        # connect servers
        for name, url in KLAVIS_SERVERS.items():
            try:
                streams = await stack.enter_async_context(streamablehttp_client(url))
                session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
                await session.initialize()
                client.sessions[name] = session
                log.info(f"Connected to {name}")
            except Exception as e:
                log.warning(f"Failed to connect to {name}: {e}")

        print("MCP Client Ready.\n")
        await client.chat_loop()

if __name__ == "__main__":
    anyio.run(main, backend="trio")
