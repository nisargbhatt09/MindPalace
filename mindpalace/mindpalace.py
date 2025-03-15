"""
Main MindPalace application class that orchestrates image processing and retrieval.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

from .config.settings import AppConfig
from .models.caption_model import CaptionModel
from .models.embedding_model import EmbeddingModel
from .database.vector_store import VectorStore


class MindPalace:
    def __init__(self, config: AppConfig):
        """Initialize MindPalace with the given configuration.

        Args:
            config (AppConfig): Application configuration
        """
        self.config = config
        
        # Initialize models
        self.caption_model = CaptionModel(config.models.caption_model_name)
        self.embedding_model = EmbeddingModel(config.models.embedding_model_name)
        
        # Initialize database
        self.vector_store = VectorStore(
            api_key=config.database.api_key,
            environment=config.database.environment,
            index_name=config.database.index_name,
            dimension=config.models.vector_dimension
        )

    def process_image(self, image_path: Path) -> Optional[str]:
        """Process a single image and store it in the database.

        Args:
            image_path (Path): Path to the image file

        Returns:
            Optional[str]: Generated caption if successful, None otherwise
        """
        # Generate caption
        caption = self.caption_model.generate_caption(image_path)
        if caption is None:
            return None

        # Generate embedding
        vector = self.embedding_model.encode_text(caption)

        # Store in database
        image_id = image_path.stem
        self.vector_store.store(image_id, caption, vector)

        return caption

    def process_directory(self) -> Dict[str, str]:
        """Process all images in the configured directory.

        Returns:
            Dict[str, str]: Dictionary mapping image IDs to their captions
        """
        results = {}
        for image_file in self.config.image_dir.glob("*"):
            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                caption = self.process_image(image_file)
                if caption:
                    results[image_file.stem] = caption
        return results

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for images using a natural language query.

        Args:
            query (str): Natural language query
            top_k (int, optional): Number of results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of matching results with metadata
        """
        query_vector = self.embedding_model.encode_text(query)
        return self.vector_store.query(query_vector, top_k) 