import asyncio
import uvicorn
from contextlib import asynccontextmanager, AsyncExitStack
from typing import Dict, Any
from fastapi import FastAPI, Request, Response
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from mcp.types import Tool
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# --- CONFIGURATION ---
UPSTREAM_SERVERS = {
    "gmail": "https://gcalendar-mcp-server.klavis.ai/mcp/?instance_id=9d9a4b34-d0c5-4b8e-b633-1aa101f57de6",
    "github": "https://strata.klavis.ai/mcp/?instance_id=df9ad3af-9eb3-4287-b2f4-acbaa5db1138",
    "linear": "https://linear-mcp-server.klavis.ai/mcp/?instance_id=8e711cd1-909a-4641-95e7-b3d5ee358110",
    "gcalendar": "https://gcalendar-mcp-server.klavis.ai/mcp/?instance_id=9d9a4b34-d0c5-4b8e-b633-1aa101f57de6",
}

upstream_sessions: Dict[str, ClientSession] = {}


# --- HELPER CLASS ---
class AlreadyHandledResponse(Response):
    """
    A special Response that does nothing.
    We return this to tell FastAPI that the MCP library has already
    managed the connection and sent the response headers.
    """

    async def __call__(self, scope, receive, send):
        pass


# --- SERVER SETUP ---
app = FastAPI()
mcp_server = Server("meta-mcp-gateway")
sse = SseServerTransport("/mcp")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🔌 Connecting to upstream Klavis servers...")
    async with AsyncExitStack() as stack:
        for name, url in UPSTREAM_SERVERS.items():
            try:
                streams = await stack.enter_async_context(streamablehttp_client(url))
                session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
                await session.initialize()
                upstream_sessions[name] = session
                print(f"✅ Connected to {name}")
            except Exception as e:
                print(f"❌ Failed to connect to {name}: {e}")
        yield
    print("🔌 Shutting down upstream connections...")


app.router.lifespan_context = lifespan


# --- MCP LOGIC ---

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    all_tools = []
    for service_name, session in upstream_sessions.items():
        try:
            result = await session.list_tools()
            for tool in result.tools:
                prefixed_name = f"{service_name}_{tool.name}"
                new_tool = Tool(
                    name=prefixed_name,
                    description=f"[{service_name.upper()}] {tool.description}",
                    inputSchema=tool.inputSchema
                )
                all_tools.append(new_tool)
        except Exception as e:
            print(f"Error fetching tools from {service_name}: {e}")
    return all_tools


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> Any:
    if "_" not in name:
        raise ValueError(f"Unknown tool format: {name}")

    service_name, actual_tool_name = name.split("_", 1)

    if service_name not in upstream_sessions:
        raise ValueError(f"Service {service_name} not found.")

    print(f"routing {name} -> {service_name} executing {actual_tool_name}")
    session = upstream_sessions[service_name]
    result = await session.call_tool(actual_tool_name, arguments)
    return result


# --- TRANSPORT ---

@app.get("/mcp")
async def handle_sse(request: Request):
    """
    1. connect_sse() starts the stream and sends headers.
    2. mcp_server.run() keeps the connection open.
    3. When done, we return AlreadyHandledResponse() so FastAPI doesn't crash.
    """
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp_server.run(
            streams[0],
            streams[1],
            mcp_server.create_initialization_options()
        )

    return AlreadyHandledResponse()


@app.post("/mcp")
async def handle_messages(request: Request):
    await sse.handle_post_message(request.scope, request.receive, request._send)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)