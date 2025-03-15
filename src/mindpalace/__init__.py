"""
MindPalace - An AI-powered image captioning and retrieval system.
"""

from .mindpalace import MindPalace
from .config.settings import AppConfig, ModelConfig, PineconeConfig

__version__ = "0.1.0"
__all__ = ["MindPalace", "AppConfig", "ModelConfig", "PineconeConfig"] 