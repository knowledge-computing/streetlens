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
        self.task_types_dict = None

    def construct_role_prompt(self):
        if not self.data_config.paper_path:
            self.data_config.system_logger.info(f"No paper input to generate role prompt")
            return
        
        with open(self.data_config.paper_path, 'r') as file:
            paper_data = json.load(file)

        abstracts = list(paper_data.values())
        self.data_config.agent_logger.info(f"Looks like we have {len(abstracts)} abstract(s). Let me step into a new role. I’m ready for it!")

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
        self.data_config.agent_logger.info(f"{role_answer_prompt}")

        return self.role_prompt

    def parse_binary_label(self,text_raw):
        # Prefer content after the last 'assistant' marker
        segment = text_raw.rsplit("assistant", 1)[-1] if "assistant" in text_raw else text_raw
        import re
        matches = re.findall(r'(?<!\d)([01])(?!\d)', segment)
        if not matches:
            matches = re.findall(r'(?<!\d)([01])(?!\d)', text_raw)
        return str(int(matches[-1]))
        
    def identify_task_type(self):
        codes_task_type = {}
        import json
       
        f_codebook = open(self.data_config.codebook_path,'r')
        codebook_raw_dict = json.load(f_codebook)
        f_codebook.close()
        
        for each_code in codebook_raw_dict.keys():
            task_prompt = f'''
                You are an annotation task classifier. 
                Given a question and its answer options, decide if the task is **perception** (holistic/qualitative scene judgment such as condition/quality/intensity ratings) or **object_detection** (presence, counting, or localization of specific object instances). 
                Rules: If it asks to rate/assess overall condition or quality (e.g., Good/Fair/Poor), label as perception. 
                If it asks to detect, count, or verify specific objects (e.g., cars, signs, pedestrians), label as object_detection.
                Question: {codebook_raw_dict[each_code]['question']}
                Answer options: {codebook_raw_dict[each_code]['answer_options']}
                Return only a single integer: 0 if perception, 1 if object_detection. 
                Do not include any words, JSON, spaces, or punctuation.
                '''
            task_type_raw = self.vlm_processor.run(task_prompt, image_path=None, max_new_tokens=1024).strip()
            task_type = self.parse_binary_label(task_type_raw)
            codes_task_type[each_code] = str(task_type)
        
        self.task_types_dict = codes_task_type
        return self.task_types_dict 

    def construct_codebook_prompt(self):
        if not self.data_config.codebook_path:
            print('No codebook input to construct codebook prompt')
            return

        self.data_config.agent_logger.info(f"I'm reading codebook...")
        with open(self.data_config.codebook_path, 'r') as file:
            codebook_dict = json.load(file)
        
        output_dict = {}
        for key, value in codebook_dict.items():
            self.data_config.agent_logger.info(f"I'm looking over the question-answer pairs for measure {key}. I'm refining the codebook prompt...")
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