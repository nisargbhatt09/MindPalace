import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
# import pinecone
from pinecone import pinecone
import numpy as np

print("Hello")

# Step 1: Initialize Models
def initialize_models():
    # Image Captioning Model (BLIP)
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # Text Embedding Model (Sentence Transformers)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    return caption_processor, caption_model, embedder

# Step 2: Generate Caption for an Image
def generate_caption(image_path, processor, model):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return None

# Step 3: Encode Text into Vector
def encode_text(text, embedder):
    return embedder.encode(text)

# Step 4: Initialize Pinecone Vector Database
def initialize_pinecone(api_key, environment, index_name):
    pinecone.init(api_key=api_key, environment=environment)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=384)  # Matches Sentence Transformer output
    return pinecone.Index(index_name)

# Step 5: Store Data in Pinecone
def store_in_database(index, image_id, caption, caption_vector):
    vector = caption_vector.tolist()
    metadata = {"caption": caption}
    index.upsert([(image_id, vector, metadata)])

# Step 6: Query the Database
def query_database(index, query_text, embedder, top_k=5):
    query_vector = embedder.encode(query_text)
    results = index.query(vector=query_vector.tolist(), top_k=top_k, include_metadata=True)
    return results


print("Hello Again")
# Main Workflow
def main():
    # Configuration
    PINECONE_API_KEY = "your_pinecone_api_key"
    PINECONE_ENVIRONMENT = "your_pinecone_environment"
    INDEX_NAME = "image-memory"
    IMAGE_DIR = "./images"  # Directory containing images
    QUERY = "Show me images of dogs playing outdoors"

    # Initialize models
    caption_processor, caption_model, embedder = initialize_models()

    # Initialize Pinecone
    index = initialize_pinecone(PINECONE_API_KEY, PINECONE_ENVIRONMENT, INDEX_NAME)

    # Process Images and Store in Database
    for image_file in os.listdir(IMAGE_DIR):
        image_path = os.path.join(IMAGE_DIR, image_file)
        image_id = os.path.splitext(image_file)[0]

        # Generate caption
        caption = generate_caption(image_path, caption_processor, caption_model)
        if caption:
            print(f"Caption for {image_file}: {caption}")

            # Encode caption into vector
            caption_vector = encode_text(caption, embedder)

            # Store in Pinecone
            store_in_database(index, image_id, caption, caption_vector)
            print(f"Stored {image_file} in database.")

    # Query the Database
    results = query_database(index, QUERY, embedder)
    print("\nQuery Results:")
    for match in results['matches']:
        print(f"Image ID: {match['id']}, Caption: {match['metadata']['caption']}, Score: {match['score']}")

if __name__ == "__main__":
    main()