import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_HOME'] = '/data/vllm'

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch

class VLMProcessor:
    def __init__(self, data_config, device):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.data_config = data_config
        self.model_name = self.data_config.model_name
        if 'hf' in self.model_name:
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, device_map=device, torch_dtype=torch.bfloat16)

    def generate(self, inputs, max_new_tokens=50):
        with torch.inference_mode():
            return self.model.generate(**inputs, max_new_tokens=max_new_tokens)