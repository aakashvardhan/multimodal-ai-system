from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

def add_qlora_adapter(model, rank=8, alpha=32, target_modules=None):
    if target_modules is None:
        # Use a more generic set of target modules
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=target_modules
    )
    return get_peft_model(model, peft_config)

# Example usage:
# phi_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
# phi_model_with_lora = add_qlora_adapter(phi_model)
