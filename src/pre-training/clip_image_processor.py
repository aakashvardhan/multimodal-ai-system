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
        self, dataset, coco_dir, embedding_file="embeddings/image_embeddings.npz"):
        if os.path.exists(embedding_file):
            print("Loading existing embeddings...")
            return np.load(embedding_file)['embeddings'], np.load(embedding_file)['image_ids']

        print("Computing and saving embeddings...")
        image_embeddings = []
        image_ids = []
        
        for entry in tqdm(dataset, desc="Processing images", unit="img"):
            image_path = os.path.join(coco_dir, "train2017", entry["image"])
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    embedding = outputs.cpu().numpy()
                    # Normalize the embedding
                    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                    image_embeddings.append(embedding.squeeze())
                    image_ids.append(entry["id"])

        image_embeddings = np.array(image_embeddings)
        image_ids = np.array(image_ids)
        print("Saving embeddings...")
        np.savez(embedding_file, embeddings=image_embeddings, image_ids=image_ids)
        return image_embeddings, image_ids

def main():
    # Example usage
    processor = ImageProcessor()
    dataset = processor.load_dataset("data/llava_instruct_150k.json")
    embeddings, image_ids = processor.process_and_save_embeddings(dataset, "data/coco")
    print(f"Processed {len(embeddings)} images")
    
    # Verify normalization
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Mean norm of embeddings: {norms.mean():.6f}")
    print(f"Std dev of norms: {norms.std():.6f}")

if __name__ == "__main__":
    main()