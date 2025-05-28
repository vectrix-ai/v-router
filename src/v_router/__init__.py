"""Vectrix Router - Unified LLM interface with automatic fallback."""

from .classes.llm import LLM, BackupModel
from .client import Client
from .providers.base import Message, Response

__all__ = [
    "Client",
    "LLM",
    "BackupModel",
    "Message",
    "Response",
]

__version__ = "0.1.0"

def main() -> None:
    print("Hello from Vectrix Router!")
