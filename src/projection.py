import torch
import torch.nn as nn

class CLIPToPhiProjection(nn.Module):
    def __init__(self, clip_embedding_dim, phi_embedding_dim):
        super().__init__()
        self.projection = nn.Linear(clip_embedding_dim, phi_embedding_dim)
    
    def forward(self, clip_embeddings):
        return self.projection(clip_embeddings)

# Example usage:
# clip_embedding_dim = 768  # Adjust based on your CLIP model
# phi_embedding_dim = 2048  # Adjust based on your Phi model
# projection = CLIPToPhiProjection(clip_embedding_dim, phi_embedding_dim)