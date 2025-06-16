import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
from types import TracebackType
from typing import Any, AsyncIterator, AsyncContextManager

from langchain_core.documents.base import Blob
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from mcp import ClientSession

from langchain_mcp_adapters.prompts import load_mcp_prompt
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.sessions import (
    Connection,
    McpHttpClientFactory,
    SSEConnection,
    StdioConnection,
    StreamableHttpConnection,
    WebsocketConnection,
    create_session,
)
from langchain_mcp_adapters.tools import load_mcp_tools

ASYNC_CONTEXT_MANAGER_ERROR = (
    "As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient cannot be used as a context manager (e.g., async with MultiServerMCPClient(...)). "
    "Instead, you can do one of the following:\n"
    "1. client = MultiServerMCPClient(...)\n"
    "   tools = await client.get_tools()\n"
    "2. client = MultiServerMCPClient(...)\n"
    "   async with client.session(server_name) as session:\n"
    "       tools = await load_mcp_tools(session)"
)


class MultiServerMCPClient:
    """Client for connecting to multiple MCP servers and loading LangChain-compatible tools, prompts and resources from them."""

    def __init__(
        self,
        connections: dict[str, Connection] | None = None,
    ) -> None:
        """Initialize a MultiServerMCPClient with MCP servers connections.

        Args:
            connections: A dictionary mapping server names to connection configurations.
                If None, no initial connections are established.

        Example: basic usage (starting a new session on each tool call)

        ```python
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            {
                "math": {
                    "command": "python",
                    # Make sure to update to the full absolute path to your math_server.py file
                    "args": ["/path/to/math_server.py"],
                    "transport": "stdio",
                },
                "weather": {
                    # make sure you start your weather server on port 8000
                    "url": "http://localhost:8000/mcp",
                    "transport": "streamable_http",
                }
            }
        )
        all_tools = await client.get_tools()
        ```

        Example: explicitly starting a session

        ```python
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        client = MultiServerMCPClient({...})
        async with client.session("math") as session:
            tools = await load_mcp_tools(session)
        ```

        Example: get all tools from all servers while keep session
        ```python
        client = MultiServerMCPClient({...})
        async with client.tools() as all_tools:
            # use all tools here
            # sessions are kept alive until exiting the context
        ```
        """
        self.connections: dict[str, Connection] = connections if connections is not None else {}

    @asynccontextmanager
    async def session(
        self,
        server_name: str,
        *,
        auto_initialize: bool = True,
    ) -> AsyncIterator[ClientSession]:
        """Connect to an MCP server and initialize a session.

        Args:
            server_name: Name to identify this server connection
            auto_initialize: Whether to automatically initialize the session

        Raises:
            ValueError: If the server name is not found in the connections

        Yields:
            An initialized ClientSession
        """
        if server_name not in self.connections:
            raise ValueError(
                f"Couldn't find a server with name '{server_name}', expected one of '{list(self.connections.keys())}'"
            )

        async with create_session(self.connections[server_name]) as session:
            if auto_initialize:
                await session.initialize()
            yield session

    @asynccontextmanager
    async def tools(self, *, server_name: str | None = None) -> AsyncIterator[list[BaseTool]]:
        """Get tools from MCP server(s) while keeping the session(s) alive.

        Args:
            server_name: Optional name of the server to get tools from.
                If None, tools from all servers will be returned (default).

        Yields:
            A list of LangChain tools
        """
        async with AsyncExitStack() as stack:
            all_tools: list[BaseTool] = []
            if server_name is not None:
                if server_name not in self.connections:
                    raise ValueError(
                        f"Couldn't find a server with name '{server_name}', expected one of '{list(self.connections.keys())}'"
                    )
                session = await stack.enter_async_context(self.session(server_name))
                tools = await load_mcp_tools(session)
                all_tools.extend(tools)
            else:
                for name in self.connections:
                    session = await stack.enter_async_context(self.session(name))
                    tools = await load_mcp_tools(session)
                    all_tools.extend(tools)

            yield all_tools

    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]:
        """Get a list of all tools from all connected servers.

        Args:
            server_name: Optional name of the server to get tools from.
                If None, all tools from all servers will be returned (default).

        NOTE: a new session will be created for each tool call

        Returns:
            A list of LangChain tools
        """
        if server_name is not None:
            if server_name not in self.connections:
                raise ValueError(
                    f"Couldn't find a server with name '{server_name}', expected one of '{list(self.connections.keys())}'"
                )
            return await load_mcp_tools(None, connection=self.connections[server_name])

        all_tools: list[BaseTool] = []
        load_mcp_tool_tasks = []
        for connection in self.connections.values():
            load_mcp_tool_task = asyncio.create_task(load_mcp_tools(None, connection=connection))
            load_mcp_tool_tasks.append(load_mcp_tool_task)
        tools_list = await asyncio.gather(*load_mcp_tool_tasks)
        for tools in tools_list:
            all_tools.extend(tools)
        return all_tools

    async def get_prompt(
        self, server_name: str, prompt_name: str, *, arguments: dict[str, Any] | None = None
    ) -> list[HumanMessage | AIMessage]:
        """Get a prompt from a given MCP server."""
        async with self.session(server_name) as session:
            prompt = await load_mcp_prompt(session, prompt_name, arguments=arguments)
            return prompt

    async def get_resources(
        self, server_name: str, *, uris: str | list[str] | None = None
    ) -> list[Blob]:
        """Get resources from a given MCP server.

        Args:
            server_name: Name of the server to get resources from
            uris: Optional resource URI or list of URIs to load. If not provided, all resources will be loaded.

        Returns:
            A list of LangChain Blobs
        """
        async with self.session(server_name) as session:
            resources = await load_mcp_resources(session, uris=uris)
            return resources

    async def __aenter__(self) -> "MultiServerMCPClient":
        raise NotImplementedError(ASYNC_CONTEXT_MANAGER_ERROR)

    def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        raise NotImplementedError(ASYNC_CONTEXT_MANAGER_ERROR)


__all__ = [
    "MultiServerMCPClient",
    "McpHttpClientFactory",
    "SSEConnection",
    "StdioConnection",
    "StreamableHttpConnection",
    "WebsocketConnection",
]
