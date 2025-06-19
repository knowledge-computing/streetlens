import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_HOME'] = '/data/vllm'

import json

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

class AutomatedPromptTuner:
    def __init__(self, data_config, vlm_processor):
        self.data_config = data_config
        self.vlm_processor = vlm_processor
        self.role_prompt = None

    def construct_role_prompt(self):
        if not self.data_config.paper_path:
            print('No paper input to generate role prompt')
            return
        
        with open(self.data_config.paper_path, 'r') as file:
            paper_data = json.load(file)

        abstracts = list(paper_data.values())
        print(f"Agent: Received {len(abstracts)} abstract(s)... Generating domain-specific role prompt...")

        abstracts = '\n\n '.join(abstracts)
        question = f"""
        You are an expert in the following fields and the author of the paper abstracts provided here: {abstracts}.\n\n
        Based on the expertise demonstrated, generate a general professional role description of yourself in one to two sentences, starting with "You are" and written in the second person.\n
        This will be used as a prompt introduction.
        """
        decoded = self.vlm_processor.run(self.vlm_processor.prepare_messages(question))
        role_prompt = decoded.split('assistant')[-1].strip()
        print(f"Agent: Here is the generated role prompt:", role_prompt)

        self.role_prompt = role_prompt
        return role_prompt

