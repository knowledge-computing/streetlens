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
        print(f"Stella: Looks like we have {len(abstracts)} abstract(s). Let me step into a new role. I’m ready for it!")

        abstracts = '\n\n '.join(abstracts)
        question = f"""
        You are an expert in the following fields and the author of the paper abstracts provided here: {abstracts}.\n\n
        Based on the expertise demonstrated, generate a general professional role description of yourself in one to two sentences, starting with "You are" and written in the second person.\n
        This will be used as a prompt introduction.
        """
        decoded = self.vlm_processor.run(self.vlm_processor.prepare_messages(question), max_new_tokens=75)
        role_prompt = decoded.split('assistant')[-1].strip()

        self.role_prompt = role_prompt

        decoded = self.vlm_processor.run(self.vlm_processor.prepare_messages(role_prompt+"Who are you?"), max_new_tokens=75)
        role_answer_prompt = decoded.split('assistant')[-1].strip()
        print(f"Stella: {role_answer_prompt}")

        return self.role_prompt

    def construct_codebook_prompt(self):
        if not self.data_config.codebook_path:
            print('No codebook input to construct codebook prompt')
            return

        print(f"Stella: I'm reading codebook...")
        with open(self.data_config.codebook_path, 'r') as file:
            codebook_dict = json.load(file)
        
        output_dict = {}
        for key, value in codebook_dict.items():
            print(f"Stella: I'm looking over the question-answer pairs for measure {key}. I'm refining the codebook prompt....")
            qa_pair = f"{{{value}}}"
            while True:
                question = f"""
                {qa_pair}\n\n

                Review the question and answer options above. Guide a vision-language model to assess environmental features in street view images using only visual input.\n

                First, write one sentence to complete the system prompt.\n

                Then, write 2–3 clear sentences for the user prompt using the EXACT SAME numeric options.\n\n

                Please answer the following using only two lines:\n
                system_prompt: <your system prompt text here>\n
                user_prompt: <your user prompt text here>

                Do NOT include any JSON, brackets, or code formatting. Just output exactly these two lines.
                """

                decoded = self.vlm_processor.run(self.vlm_processor.prepare_messages(question), max_new_tokens=2000)
                response = decoded.split('assistant')[-1].strip()
                try:
                    response_dict = {}
                    for line in response.strip().splitlines():
                        line = line.strip()
                        if not line or ':' not in line:
                            continue
                        k, v = line.split(":", 1)
                        response_dict[k.strip()] = v.strip()

                    output_dict[key] = response_dict
                    if self.role_prompt is not None:
                        output_dict[key]['system_prompt'] = self.role_prompt + output_dict[key]['system_prompt']
                    output_dict[key]['user_prompt'] = output_dict[key]['user_prompt'] + " DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION."
                    break
                except Exception as e:
                    print(e)
                    continue

        self.codebook_prompt_dict = output_dict
        return self.codebook_prompt_dict