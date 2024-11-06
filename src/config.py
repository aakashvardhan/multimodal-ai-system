from dataclasses import dataclass

@dataclass
class Config:
    # Model configuration
    clip_embedding_dim: int = 512
    phi_embedding_dim: int = 3072
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    num_projection_layers: int = 6
    
    # Memory optimization settings
    batch_size: int = 4  # Reduced batch size
    gradient_accumulation_steps: int = 8
    use_gradient_checkpointing: bool = True
    mixed_precision_dtype: str = "bfloat16"
    load_in_8bit: bool = True
    max_grad_norm: float = 1.0
    
    # Optimizer memory settings
    optimizer_states_in_cpu: bool = True
    empty_cuda_cache_frequency: int = 1
    
    # Training configuration
    num_epochs: int = 5
    learning_rate: float = 1e-4
    max_lr: float = 2e-4
    
    # Dataset settings
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    max_length: int = 128
    
    # Paths
    dataset_path: str = "data/llava_instruct_150k.json"
    image_embeddings_path: str = "embeddings/clip_embeddings.pt"
    checkpoint_dir: str = "src/checkpoints"
    save_every: int = 1