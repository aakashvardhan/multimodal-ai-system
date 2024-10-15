import os
import torch
import numpy as np
from PIL import Image
import clip
from datasets import load_dataset


class ImageProcessor:
    def __init__(self, clip_model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(
            clip_model_name, device=self.device
        )

    def load_dataset(self, json_file_path):
        return load_dataset("json", data_files=json_file_path, split="train")

    def process_and_save_embeddings(
        self, dataset, coco_dir, embedding_file="image_embeddings.npy"
    ):
        if os.path.exists(embedding_file):
            print("Loading existing embeddings...")
            return np.load(embedding_file)

        print("Computing and saving embeddings...")
        image_embeddings = []
        for entry in dataset:
            image_path = os.path.join(coco_dir, "train2017", entry["image"])
            if os.path.exists(image_path):
                img = (
                    self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
                )
                with torch.no_grad():
                    embedding = self.clip_model.encode_image(img)
                    image_embeddings.append(embedding.cpu().numpy())

        image_embeddings = np.array(image_embeddings)
        np.save(embedding_file, image_embeddings)
        return image_embeddings


def main():
    # Example usage
    processor = ImageProcessor()
    dataset = processor.load_dataset("data/llava_instruct_150k.json")
    embeddings = processor.process_and_save_embeddings(dataset, "data/coco")
    print(f"Processed {len(embeddings)} images")


if __name__ == "__main__":
    main()
