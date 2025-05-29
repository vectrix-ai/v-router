"""Vectrix Router - Unified LLM interface with automatic fallback."""

from dotenv import load_dotenv

from v_router.classes.llm import LLM, BackupModel
from v_router.client import Client
from v_router.logger import setup_logger
from v_router.providers.base import Message, Response

load_dotenv()


__all__ = [
    "Client",
    "LLM",
    "BackupModel",
    "Message",
    "Response",
    "setup_logger",
]
