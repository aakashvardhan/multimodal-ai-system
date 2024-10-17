import json
from datasets import load_dataset
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageCaptionPreprocessor:
    def __init__(self, dataset_path: str):
        try:
            self.dataset = load_dataset("json", data_files=dataset_path)
            self.train_data = self.dataset['train']
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def extract_captions(self) -> List[Dict[str, str]]:
        captions = []
        for item in self.train_data:
            conversation = item.get('conversations', [])
            image_id = item.get('id', '')
            
            caption = self._extract_caption_from_conversation(conversation)
            
            if caption and image_id:
                captions.append({
                    'image_id': image_id,
                    'caption': caption
                })
        
        logger.info(f"Extracted {len(captions)} captions")
        return captions

    def _extract_caption_from_conversation(self, conversation: List[Dict[str, str]]) -> Optional[str]:
        for turn in conversation:
            if turn.get('from') == 'human':
                caption = turn.get('value', '').lstrip("Human: ").strip()
                return caption if caption else None
        return None

    def save_captions_to_file(self, output_file: str):
        captions = self.extract_captions()
        try:
            with open(output_file, 'w') as f:
                json.dump(captions, f, indent=2)
            logger.info(f"Saved {len(captions)} captions to {output_file}")
        except IOError as e:
            logger.error(f"Failed to save captions to file: {e}")

    def get_sample_captions(self, n: int = 5) -> List[Dict[str, str]]:
        captions = self.extract_captions()
        return captions[:min(n, len(captions))]

# Usage example
if __name__ == "__main__":
    try:
        preprocessor = ImageCaptionPreprocessor(dataset_path="data/llava_instruct_150k.json")
        
        # Get a few sample captions
        samples = preprocessor.get_sample_captions(5)
        logger.info("Sample captions:")
        for sample in samples:
            logger.info(f"Image ID: {sample['image_id']}")
            logger.info(f"Caption: {sample['caption']}\n")
        
        # Save all captions to a file
        preprocessor.save_captions_to_file(output_file="data/preprocessed_captions.json")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
