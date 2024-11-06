import os
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class CLIPEmbeddingGenerator:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def load_dataset(self, json_path):
        """Load dataset from JSON file"""
        return load_dataset("json", data_files=json_path, split="train")


class ImageDataset(Dataset):
    def __init__(self, dataset, image_dir, processor):
        self.dataset = dataset
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Remove any reference to 'content' in the path since it's already in image_dir
        image_path = os.path.join(self.image_dir, "train2017", item["image"])
        try:
            image = Image.open(image_path)
            # Process image using the CLIP processor
            processed = self.processor(images=image, return_tensors="pt")
            # Remove the batch dimension that the processor adds
            image_tensor = processed["pixel_values"].squeeze(0)
            return image_tensor, item["id"]
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            raise


def main(
    json_path="data/llava_instruct_150k.json",
    image_dir="/content/coco",
    output_path="clip_embeddings.pt",
    batch_size=32,
):
    # Initialize CLIP processor
    generator = CLIPEmbeddingGenerator()
    dataset = generator.load_dataset(json_path)

    # Create dataset and dataloader
    image_dataset = ImageDataset(dataset, image_dir, generator.processor)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Process batches and store embeddings
    embeddings = {}
    with torch.no_grad():
        for batch_images, batch_ids in tqdm(dataloader, desc="Processing images"):
            # Move batch to device
            batch_images = batch_images.to(generator.device)

            # Generate embeddings using get_image_features instead of encode_image
            batch_features = generator.model.get_image_features(batch_images)

            # Store embeddings
            for idx, image_id in enumerate(batch_ids):
                embeddings[image_id] = batch_features[idx].cpu().numpy()

    # Save embeddings
    torch.save(embeddings, output_path)
    print(f"Embeddings saved to {output_path}")


if __name__ == "__main__":
    main()
