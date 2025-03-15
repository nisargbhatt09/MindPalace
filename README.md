
![1*4KghD9iFGGzUVySCpU7E5A](https://github.com/user-attachments/assets/cc41da9c-69f5-4644-8100-ec19d5e02094)

# Image Caption Memory

A sophisticated image processing and retrieval system that combines state-of-the-art image captioning with semantic search capabilities. The system automatically generates natural language descriptions for images and enables semantic search through a vector database.

## Features

- 🖼️ Automatic image captioning using BLIP (Bootstrapping Language-Image Pre-training)
- 🔍 Semantic search capabilities for finding similar images
- 💾 Vector database storage using Pinecone
- 🔄 Real-time processing and retrieval
- 📊 Similarity scoring for search results

## Technical Stack

- **Image Captioning**: Salesforce BLIP (blip-image-captioning-large)
- **Text Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: Pinecone
- **Image Processing**: PIL (Python Imaging Library)
- **Deep Learning Framework**: PyTorch

## Prerequisites

- Python 3.7+
- Pinecone API Key
- Sufficient storage for image processing

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd image-caption-memory
```

2. Install required packages:
```bash
pip install torch transformers sentence-transformers pinecone-client Pillow
```

3. Configure environment variables:
- Set up your Pinecone API key
- Configure your Pinecone environment
- Set up your index name

## Usage

1. Place your images in the `./images` directory

2. Update the configuration in `main()`:
```python
PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_ENVIRONMENT = "your_pinecone_environment"
INDEX_NAME = "image-memory"
```

3. Run the script:
```bash
python image_caption_memory.py
```

## How It Works

1. **Image Processing**: The system processes images using BLIP to generate natural language captions
2. **Vector Embedding**: Captions are converted into vector embeddings using Sentence Transformers
3. **Storage**: Embeddings and metadata are stored in Pinecone vector database
4. **Retrieval**: Semantic search queries return the most relevant images based on caption similarity

## Example Output

```
Caption for image1.jpg: A dog playing with a frisbee in the park
Stored image1.jpg in database.

Query Results:
Image ID: image1, Caption: A dog playing with a frisbee in the park, Score: 0.89
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Acknowledgments

- Salesforce for the BLIP model
- Sentence Transformers team
- Pinecone for vector database services 

# Install dependencies
pip install -r requirements.txt

# Process images in a directory
python src/main.py --image-dir ./images

# Search for images
python src/main.py --query "dogs playing outdoors" --top-k 5 
