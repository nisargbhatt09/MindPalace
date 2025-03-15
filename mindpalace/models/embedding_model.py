"""
Text embedding model implementation using Sentence Transformers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str):
        """Initialize the Sentence Transformer model.

        Args:
            model_name (str): Name of the pretrained model to use
        """
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text into a vector embedding.

        Args:
            text (str): Text to encode

        Returns:
            np.ndarray: Vector embedding of the text
        """
        return self.model.encode(text) 