"""Vectrix Router - Unified LLM interface with automatic fallback."""

from dotenv import load_dotenv

from .classes.llm import LLM, BackupModel
from .client import Client
from .logger import setup_logger
from .providers.base import Message, Response

load_dotenv()


__all__ = [
    "Client",
    "LLM",
    "BackupModel",
    "Message",
    "Response",
    "setup_logger",
]
