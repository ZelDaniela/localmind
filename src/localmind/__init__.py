"""LocalMind - Persistent memory system for local AI agents."""

import os

os.environ["HF_HUB_DISABLE_DOWNLOAD_WARNINGS"] = "1"

__version__ = "0.1.0"

from localmind.memory import MemoryStore

__all__ = ["MemoryStore", "__version__"]