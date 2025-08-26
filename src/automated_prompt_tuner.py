import os
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
        Based on the expertise demonstrated, generate a general professional role description of yourself in one sentence, starting with "You are" and written in the second person.\n
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
        segment = text_raw.rsplit("assistant", 1)[-1] if "assistant" in text_raw else text_raw
        import re
        matches = re.findall(r'(?<!\d)([01])(?!\d)', segment)
        if not matches:
            matches = re.findall(r'(?<!\d)([01])(?!\d)', text_raw)
        return str(int(matches[-1]))

    def task_type_to_prompt(self,task_type_code):
        if task_type_code == '0':
            task_prompt = '''Focus on holistic visual cues rather than counting specific objects.'''
        else:
            task_prompt = '''Detect the specified object(s) strictly from visible evidence. Report only presence/absence or counts as required.'''
        return task_prompt


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
        self.identify_task_type()
        task_types_dict = self.task_types_dict
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
                question = f"""Instruction: Rewrite the question as a clear, self-contained sentence, prefixed with "Question:". Then, rewrite each answer option as a full sentence explaining the meaning, starting with its number. Keep all numbers and meaning intact. Output plain text only, one sentence per line.

                Question: {codebook_dict[key]['question']} Answer options: {codebook_dict[key]['answer_options']}"""
                decoded = self.vlm_processor.run(self.vlm_processor.prepare_messages(question), max_new_tokens=2000)
                response = decoded.split('assistant')[-1].strip()
                try:
                    response_dict = {}
                    output_dict[key] = response_dict
                    if self.role_prompt is not None:
                        output_dict[key]['system_prompt'] = self.role_prompt
                    task_prompt = self.task_type_to_prompt(task_types_dict[key])
                    output_dict[key]['user_prompt'] = task_prompt + " " + response.strip() + " DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION."
                    break
                except Exception as e:
                    print(e)
                    continue

        self.codebook_prompt_dict = output_dict
        return self.codebook_prompt_dict