"""
Configuration settings for the MindPalace application.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    caption_model_name: str = "Salesforce/blip-image-captioning-large"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    vector_dimension: int = 384


@dataclass
class PineconeConfig:
    api_key: str = os.getenv("PINECONE_API_KEY", "your_pinecone_api_key")
    environment: str = os.getenv("PINECONE_ENVIRONMENT", "your_pinecone_environment")
    index_name: str = os.getenv("PINECONE_INDEX_NAME", "image-memory")


@dataclass
class AppConfig:
    image_dir: Path = Path("./images")
    models: ModelConfig = ModelConfig()
    database: PineconeConfig = PineconeConfig() 