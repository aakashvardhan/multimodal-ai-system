import os
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

class ImageProcessor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def load_dataset(self, json_file_path):
        return load_dataset("json", data_files=json_file_path, split="train")

    def process_and_save_embeddings(
        self, dataset, coco_dir, embedding_file="image_embeddings.npz"
    ):
        if os.path.exists(embedding_file):
            print("Loading existing embeddings...")
            return np.load(embedding_file)['embeddings']

        print("Computing and saving embeddings...")
        image_embeddings = []
        image_ids = []
        
        # Create a tqdm progress bar
        for entry in tqdm(dataset, desc="Processing images", unit="img"):
            image_path = os.path.join(coco_dir, "train2017", entry["image"])
            if os.path.exists(image_path):
                image = Image.open(image_path)
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    embedding = outputs.cpu().numpy()
                    image_embeddings.append(embedding)
                    image_ids.append(entry["image"])

        image_embeddings = np.array(image_embeddings)
        print("Saving embeddings...")
        np.savez(embedding_file, embeddings=image_embeddings, image_ids=image_ids)
        return image_embeddings

def main():
    # Example usage
    processor = ImageProcessor()
    dataset = processor.load_dataset("data/llava_instruct_150k.json")
    embeddings = processor.process_and_save_embeddings(dataset, "data/coco")
    print(f"Processed {len(embeddings)} images")

if __name__ == "__main__":
    main()
