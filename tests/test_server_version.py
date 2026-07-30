"""The version the server advertises over MCP."""
import asyncio
from importlib.metadata import version as distribution_version

from fastmcp import Client

from deep_zotero import __version__
from deep_zotero.server import mcp


def server_info():
    """The serverInfo a client receives from the initialize handshake."""
    async def initialize():
        async with Client(mcp) as client:
            return client.initialize_result.serverInfo

    return asyncio.run(initialize())


def test_package_version_comes_from_the_installed_distribution():
    assert __version__ == distribution_version("deep-zotero")


def test_initialize_reports_the_package_version():
    info = server_info()
    assert info.name == "deep-zotero"
    assert info.version == __version__
