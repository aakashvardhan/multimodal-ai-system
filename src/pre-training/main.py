import logging
import os
from typing import Dict, Optional
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from datasets import load_dataset
from image_captions_dataset import ImageCaptionDataset
from model import load_and_prepare_model
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import Config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

config = Config()

# Load dataset
dataset = load_dataset(
    "json", data_files="data/llava_instruct_150k.json", split="train"
)

tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
tokenizer.pad_token = (
    tokenizer.unk_token
)  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = "right"


# Create dataset
dataset = ImageCaptionDataset(dataset, tokenizer, config.max_length)

# Create dataloader
train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# Initialize model
model = load_and_prepare_model(config, logger)

# Optimizer
optimizer = AdamW(model.parameters(), lr=config.learning_rate)

# Loss function
criterion = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training loop
for epoch in range(config.num_epochs):
    total_loss = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{config.num_epochs}",
        total=len(train_loader),
    )
    model.train()
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        # Unpack the tuple returned by dataset
        image_ids, input_ids, target_ids = batch
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Create attention mask that includes both image and text tokens
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        # Add 1 for the image token position
        full_attention_mask = torch.ones((attention_mask.shape[0], 1), device=device)
        full_attention_mask = torch.cat([full_attention_mask, attention_mask], dim=1)

        outputs = model(image_ids, input_ids, attention_mask=full_attention_mask)
        
        text_token_logits = outputs.logits[:, 1:, :]  # Start from index 1 to skip separator tokens

        # Flatten the logits and target sequence for loss calculation
        text_token_logits_flat = text_token_logits.reshape(-1, text_token_logits.size(-1))
        target_ids_flat = target_ids.reshape(-1)

        # Calculate loss over the text token sequence
        loss = criterion(text_token_logits_flat, target_ids_flat)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch+1}/{config.num_epochs} - Average Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), f"checkpoints/model_{epoch+1}.pth")
    print(f"Model saved to checkpoints/model_{epoch+1}.pth")