import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from .projection import CLIPToPhiProjection
from .qlora_adapter import add_qlora_adapter

class MultimodalPhiModel(nn.Module):
    def __init__(self, phi_model_name, clip_embedding_dim, phi_embedding_dim):
        super().__init__()
        self.phi_model = AutoModelForCausalLM.from_pretrained(phi_model_name, trust_remote_code=True)
        self.projection = CLIPToPhiProjection(clip_embedding_dim, phi_embedding_dim)
        
        # Get the actual module names from the model
        target_modules = [name for name, _ in self.phi_model.named_modules() if any(keyword in name for keyword in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"])]
        
        self.phi_model_with_lora = add_qlora_adapter(self.phi_model, target_modules=target_modules)
    
    def forward(self, input_ids, attention_mask, clip_embeddings):
        projected_embeddings = self.projection(clip_embeddings)
        # Combine projected embeddings with text input
        # This is a simplified example; you may need to adjust based on Phi's architecture
        combined_input = torch.cat([projected_embeddings, self.phi_model.get_input_embeddings()(input_ids)], dim=1)
        outputs = self.phi_model_with_lora(inputs_embeds=combined_input, attention_mask=attention_mask)
        return outputs

# Example usage:
# model = MultimodalPhiModel("microsoft/phi-1_5", clip_embedding_dim=768, phi_embedding_dim=2048)
