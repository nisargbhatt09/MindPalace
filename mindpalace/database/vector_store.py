"""
Vector database interface using Pinecone.
"""

from typing import Dict, List, Any
import numpy as np
from pinecone import Pinecone, Index


class VectorStore:
    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int):
        """Initialize the Pinecone vector database.

        Args:
            api_key (str): Pinecone API key
            environment (str): Pinecone environment
            index_name (str): Name of the index to use
            dimension (int): Dimension of the vectors to store
        """
        self.pc = Pinecone(api_key=api_key)
        
        # Create index if it doesn't exist
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
        
        self.index = self.pc.Index(index_name)

    def store(self, image_id: str, caption: str, vector: np.ndarray) -> None:
        """Store an image's caption and vector in the database.

        Args:
            image_id (str): Unique identifier for the image
            caption (str): Generated caption for the image
            vector (np.ndarray): Vector embedding of the caption
        """
        metadata = {"caption": caption}
        self.index.upsert([(image_id, vector.tolist(), metadata)])

    def query(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the database for similar vectors.

        Args:
            query_vector (np.ndarray): Vector to search for
            top_k (int, optional): Number of results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of matching results with metadata
        """
        results = self.index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        return results['matches'] 