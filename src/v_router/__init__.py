"""Vectrix Router - Unified LLM interface with automatic fallback."""

from dotenv import load_dotenv

from v_router.classes.llm import LLM, BackupModel
from v_router.classes.messages import HumanMessage, SystemMessage, ToolMessage
from v_router.classes.response import AIMessage
from v_router.client import Client
from v_router.logger import setup_logger

load_dotenv()


__all__ = [
    "Client",
    "LLM",
    "BackupModel",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "ToolMessage",
    "setup_logger",
]
