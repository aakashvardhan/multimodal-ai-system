from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

def add_qlora_adapter(model, rank=8, alpha=32):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.1
    )
    return get_peft_model(model, peft_config)

# Example usage:
# phi_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
# phi_model_with_lora = add_qlora_adapter(phi_model)