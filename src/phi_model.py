from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

class PhiModel:
    def __init__(self, model_config, peft_config, bnb_config):
        self.model_config = model_config
        self.peft_config = peft_config
        self.bnb_config = bnb_config
        self.model = None
        self.tokenizer = None

    def setup(self):
        self._setup_bnb_config()
        self._load_model()
        self._prepare_model_for_training()
        self._setup_peft()
        self._load_tokenizer()
        return self.model, self.tokenizer

    def _setup_bnb_config(self):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.bnb_config['load_in_4bit'],
            bnb_4bit_quant_type=self.bnb_config['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=self.bnb_config['bnb_4bit_compute_dtype'],
            bnb_4bit_use_double_quant=self.bnb_config['bnb_4bit_use_double_quant'],
        )

    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config['model_name'],
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=self.model_config['trust_remote_code'],
            torch_dtype=self.model_config['torch_dtype'],
        )
        self.model.config.use_cache = self.model_config['use_cache']
        self.model.config.pretraining_tp = 1

    def _prepare_model_for_training(self):
        self.model = prepare_model_for_kbit_training(self.model)

    def _setup_peft(self):
        peft_config = LoraConfig(**self.peft_config)
        self.model = get_peft_model(self.model, peft_config)

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['model_name'], 
            trust_remote_code=self.model_config['trust_remote_code']
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

# Usage:
# model_setup = ModelSetup(model_config, peft_config, bnb_config)
# model, tokenizer = model_setup.setup()

