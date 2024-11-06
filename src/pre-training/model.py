import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import gc
import numpy as np
from torch.cuda.amp import autocast
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Projections(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output = nn.Linear(config.clip_embedding_dim, config.phi_embedding_dim)
        self.norm = nn.LayerNorm(config.phi_embedding_dim)
        self.projection_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.phi_embedding_dim, config.phi_embedding_dim),
                    nn.GELU(),
                    nn.Linear(config.phi_embedding_dim, config.phi_embedding_dim),
                )
                for _ in range(config.num_projection_layers)
            ]
        )

    def forward(self, x):
        x = self.output(x)
        x = self.norm(x)
        for layer in self.projection_layers:
            residual = x
            x = layer(x) + residual
        return x


class ClipPhi3Model(nn.Module):
    def __init__(self, config, logger):
        super().__init__()

        # Load Phi model with memory optimizations
        self.phi = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=config.load_in_8bit,
        )

        # Enable gradient checkpointing
        if config.use_gradient_checkpointing:
            self.phi.gradient_checkpointing_enable()

        # Initialize projections
        self.projections = Projections(config)
        self.projections.to(torch.bfloat16)

        # Load CLIP embeddings efficiently
        self.load_clip_embeddings(config.image_embeddings_path, logger)

        # Freeze Phi model weights
        for param in self.phi.parameters():
            param.requires_grad = False

    def load_clip_embeddings(self, embeddings_path, logger):
        try:
            # Load embeddings with memory-efficient approach
            self.image_embeddings = {}
            chunk = torch.load(embeddings_path)

            for k, v in chunk.items():
                if isinstance(v, np.ndarray):
                    self.image_embeddings[k] = torch.from_numpy(v).to(torch.bfloat16)
                elif isinstance(v, torch.Tensor):
                    self.image_embeddings[k] = v.to(torch.bfloat16)
                else:
                    self.image_embeddings[k] = v

            logger.info(f"Loaded {len(self.image_embeddings)} embeddings")
            del chunk
            torch.cuda.empty_cache()

        except Exception as e:
            raise Exception(f"Error loading CLIP embeddings: {str(e)}")

    def get_image_embedding(self, image_id):
        if image_id not in self.image_embeddings:
            raise KeyError(f"No embedding found for image ID: {image_id}")
        return self.image_embeddings[image_id].to(device)

    def forward(self, image_ids, input_ids, attention_mask=None):
        with autocast(dtype=torch.bfloat16):
            # Get text embeddings
            text_embeds = self.phi.get_input_embeddings()(input_ids.to(device))

            # Process image embeddings in chunks if needed
            image_embeds = []
            chunk_size = 16  # Adjust based on memory constraints
            for i in range(0, len(image_ids), chunk_size):
                chunk_ids = image_ids[i : i + chunk_size]
                chunk_embeds = torch.stack(
                    [self.get_image_embedding(id) for id in chunk_ids]
                )
                image_embeds.append(chunk_embeds)

            image_embeds = torch.cat(image_embeds, dim=0).unsqueeze(1)

            # Project image embeddings
            projected_image_embeds = self.projections(image_embeds)

            # Combine embeddings
            combined_embeds = torch.cat([projected_image_embeds, text_embeds], dim=1)

            # Generate outputs
            outputs = self.phi(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask if attention_mask is not None else None,
            )

            return outputs
