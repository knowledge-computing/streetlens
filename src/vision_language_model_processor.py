import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_HOME'] = '/data/vllm'

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch

class VLMProcessor:
    def __init__(self, data_config, model_name, device):
        self.data_config = data_config
        self.model_name = model_name
        self.load_model(device)

    def load_model(self, device):
        if 'hf' in self.model_name:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, device_map=device, torch_dtype=torch.bfloat16)