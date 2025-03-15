"""
Configuration settings for the MindPalace application.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    caption_model_name: str = "Salesforce/blip-image-captioning-large"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    vector_dimension: int = 384


@dataclass
class PineconeConfig:
    api_key: str = "your_pinecone_api_key"  # Should be loaded from environment variable
    environment: str = "your_pinecone_environment"
    index_name: str = "image-memory"


@dataclass
class AppConfig:
    image_dir: Path = Path("./images")
    models: ModelConfig = ModelConfig()
    database: PineconeConfig = PineconeConfig() 