"""
Command-line interface for MindPalace.
"""

import argparse
import os
from pathlib import Path

from mindpalace import MindPalace, AppConfig


def main():
    parser = argparse.ArgumentParser(description="MindPalace - Image Caption Memory System")
    parser.add_argument("--image-dir", type=str, default="./images",
                       help="Directory containing images to process")
    parser.add_argument("--query", type=str,
                       help="Natural language query to search for images")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of results to return for search")
    
    args = parser.parse_args()

    # Load configuration
    config = AppConfig()
    config.image_dir = Path(args.image_dir)
    
    # Initialize MindPalace
    mind_palace = MindPalace(config)

    if args.query:
        # Search mode
        results = mind_palace.search(args.query, args.top_k)
        print(f"\nSearch results for query: {args.query}")
        print("-" * 50)
        for result in results:
            print(f"Image: {result['id']}")
            print(f"Caption: {result['metadata']['caption']}")
            print(f"Similarity Score: {result['score']:.3f}")
            print("-" * 50)
    else:
        # Processing mode
        print(f"Processing images in {config.image_dir}")
        results = mind_palace.process_directory()
        print("\nProcessed Images:")
        print("-" * 50)
        for image_id, caption in results.items():
            print(f"Image: {image_id}")
            print(f"Caption: {caption}")
            print("-" * 50)


if __name__ == "__main__":
    main() 