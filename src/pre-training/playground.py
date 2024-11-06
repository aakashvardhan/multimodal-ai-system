from torch.utils.data import DataLoader
from datasets import load_dataset
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from image_captions_dataset import ImageCaptionDataset
from config import Config

config = Config()

dataset = load_dataset("json", data_files="data/llava_instruct_150k.json", split="train")

dataset = ImageCaptionDataset(dataset, tokenizer=AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True))

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(next(iter(dataloader)))
