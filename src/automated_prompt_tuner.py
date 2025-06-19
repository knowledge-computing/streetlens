import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_HOME'] = '/data/vllm'

import json

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig


class AutomatedPromptTuner:
    def __init__(self, data_config, dtype=torch.bfloat16):
        self.data_config = data_config
        self.processor = AutoProcessor.from_pretrained(self.data_config.model_name)
        self.dtype = dtype

    def prepare_messages(self, text, image_path=None):
        content = [{"type": "text", "text": text}]
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at: {image_path}")
            image = Image.open(image_path).convert("RGB")
            content.insert(0, {"type": "image", "image": image})

        return [{"role": "user", "content": content}]

    def run(self, messages, vlm, max_new_tokens=75):
        inputs = self.processor.apply_chat_template(
            messages,
            padding=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(vlm.model.device, dtype=self.dtype)

        with torch.inference_mode():
            outputs = vlm.generate(inputs, max_new_tokens=max_new_tokens)

        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return decoded.split("assistant")[-1].strip()

    def construct_role_prompt(self):
        if not self.data_config.paper_path:
            print('No paper input to generate role prompt')
            return
        
        with open(self.data_config.paper_path, 'r') as file:
            paper_data = json.load(file)

        abstracts = list(paper_data.values())
        abstracts = '\n\n '.join(abstracts)
        question = f"""
        You are an expert in the following fields and the author of the paper abstracts provided here: {abstracts}.\n\n
        Based on the expertise demonstrated, generate a general professional role description of yourself in one to two sentences, starting with "You are" and written in the second person.\n
        This will be used as a prompt introduction.
        """
        return question

