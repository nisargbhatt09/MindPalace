"""
Image captioning model implementation using BLIP.
"""

from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Optional


class CaptionModel:
    def __init__(self, model_name: str):
        """Initialize the BLIP image captioning model.

        Args:
            model_name (str): Name of the pretrained model to use
        """
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

    def generate_caption(self, image_path: Path) -> Optional[str]:
        """Generate a caption for the given image.

        Args:
            image_path (Path): Path to the image file

        Returns:
            Optional[str]: Generated caption or None if generation fails
        """
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(image, return_tensors="pt")
            out = self.model.generate(**inputs, max_length=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return None 