"""A streaming LLM chat site that works with any OpenAI-compatible API or Anthropic."""

from .app import create_app

__all__ = ["create_app"]
__version__ = "2.0.0"
