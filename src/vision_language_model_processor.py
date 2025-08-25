import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_HOME'] = '/data/vllm'

import json
import pandas as pd

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch


class VLMProcessor:
    def __init__(self, data_config, device, dtype=torch.bfloat16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.data_config = data_config
        self.model_name = self.data_config.model_name
        self.dtype = dtype

        if 'hf' in self.model_name:
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, device_map=self.device, torch_dtype=self.dtype)
            self.processor = AutoProcessor.from_pretrained(self.data_config.model_name)

    def prepare_messages(self, text, image_path=None, image_url=None):
        content = [{"type": "text", "text": text}]
        if image_path:
            content.insert(0, {"type": "image", "image": self._load_image_tensor(image_path)})
        elif image_url:
            content.insert(0, {"type": "image", "url": image_url})

        return [{"role": "user", "content": content}]

    def _load_image_tensor(self, path):
        from PIL import Image
        image = Image.open(path).convert("RGB")
        return image

    def run(self, text, image_path=None, image_url=None, max_new_tokens=1000):
        messages = self.prepare_messages(text, image_path, image_url)

        inputs = self.processor.apply_chat_template(
            messages,
            padding=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=self.dtype)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.processor.tokenizer.eos_token_id)

        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

    def _find_majority(self, score_list):
        from collections import Counter
        if not score_list:
            return 99
        return Counter(score_list).most_common(1)[0][0]

    def _aggregate_scores(self, score_list, class_dict):
        if len(class_dict) <= 2:
            return 1 if sum(score_list) > 0 else 0
        return self._find_majority(score_list)

    def extract_dict_from_response(self, response_str):
        import re
        if isinstance(response_str, dict):
            return response_str
        pattern = r"\{[^{}]*'score'\s*:\s*\d+[^{}]*'reason'\s*:\s*'[^']*'[^{}]*\}"
        match = re.search(pattern, response_str, re.DOTALL)
        if match:
            return match.group()
        return None

    def _run_for_score(self, prompt, image_path, valid_scores, max_retries=5):
        import ast
        attempts = 0
        self.data_config.agent_logger.info(f"I got the image from {image_path}. I'm starting annotation now...")
        while attempts < max_retries:
            try:            
                decoded_str = self.run(prompt, image_path=image_path, max_new_tokens=1024).strip()
                decoded_output = self.extract_dict_from_response(decoded_str)
                decoded_output = ast.literal_eval(decoded_output)
                if not isinstance(decoded_output, dict):
                    raise ValueError("Output is not a dictionary.")
                if 'score' not in decoded_output or 'reason' not in decoded_output:
                    raise KeyError("Missing 'score' or 'reason'")
                score = int(decoded_output['score'])
                if score not in valid_scores:
                    raise ValueError("Score not in valid range")

                self.data_config.agent_logger.info(f"My annotation is {score}")
                self.data_config.agent_logger.info(f"{decoded_output['reason'][0] + decoded_output['reason'][1:]}\n")
                return score , str(decoded_output['reason'][0] + decoded_output['reason'][1:])
            except Exception as e:
                # print(f"Invalid response, retrying ({attempts + 1}/{max_retries}): {e}")
                attempts += 1
                
        # print("Maximum number of retries reached. Returning fallback score 99.")
        return 99 , ""

    def generate_agent_anno_file(self, output_dict, annotation_path, agent_annotation_path):
        add_cols_from_anno =  ['Block Face ID:','Census Tract ID:','Fully Matched','Street Block Faces on:','Boundary Streets:','Picture Date (mm/year)']
        df = pd.read_csv(annotation_path)
        df.columns = df.columns.str.strip()
        new_rows = []
        reason_dict = {}

        for block_face, scores in output_dict.items():
            block_id, direction = block_face.split('/')
        
            row = {
                'Street Block ID': block_id.strip(),
                'Direction of Target Block Face': direction.strip()
            }
            for _, anno_row in df.iterrows():
                if str(block_id).strip() in str(anno_row['Street Block ID']).strip() and str(anno_row['Direction of Target Block Face']).strip() == direction.strip():
                    for col in add_cols_from_anno:
                        row[col] = anno_row[col]
                    break  
            reason_entry = {}
            for target_code, value in scores.items():
                score = value[0]
                reasons = value[1]
                row[target_code] = score
                row[target_code] = score
                reason_entry[target_code] = f"{reasons}"
                # if isinstance(reasons, list):
                #     reason_str_list = [f"{k}: {v}" for r in reasons for k, v in r.items()]
                #     row[f"{target_code}_Reason"] = " / ".join(reason_str_list)
                # else:
                #     row[f"{target_code}_Reason"] = str(reasons)
            new_rows.append(row)
            reason_dict[f"{block_id.strip()}/{direction.strip()}"] = reason_entry

        df_agent = pd.DataFrame(new_rows)

        df_agent.to_csv(agent_annotation_path, index=False)

        reason_json_path = agent_annotation_path.replace(".csv", "_reason.json")
        with open(reason_json_path, "w", encoding="utf-8") as f:
            json.dump(reason_dict, f, indent=1, ensure_ascii=False)
        return df_agent

    def generate_annotation(self):
        results = {}
        image_dir = self.data_config.image_dir
        streetblock_ids = self.data_config.street_block_id
        role_prompt = self.data_config.role_prompt
        codebook = self.data_config.codebook_prompt_dict
        annotation_path = self.data_config.annotation_path
        agent_annotation_path = self.data_config.agent_annotation_path

        class_dict = {
            'Decay 2': [1, 2, 3, 99],
            'Decay 1': [1, 2, 3, 99],
            'SS4': [0, 1],
            'SS5': [0, 1],
            'Disorder 3': [0, 1]
        }

        target_codes = list(codebook.keys())
        target_task_types = self.data_config.task_types_dict
        # target_task_types = self.get_task_type(target_codes)
        for block_id in streetblock_ids:
            block_path = os.path.join(image_dir, block_id)
            for direction in os.listdir(block_path):
                dir_path = os.path.join(block_path, direction)
                image_score_dict = {}

                for target_code in target_codes:
                    self.data_config.agent_logger.info(f"Jumping into measure {target_code}...\n")
                    if target_task_types[target_code] == '0':
                        task_prompt = '''Assess the overall scene condition/quality from the provided inputs. 
                        Focus on holistic visual cues rather than counting specific objects   
                        '''
                    else:
                        task_prompt = '''
                        Detect the specified object(s) strictly from visible evidence. 
                        Report only presence/absence or counts as required.
                        Do not rate overall condition or add qualitative judgments.
                        '''
                    valid_scores = class_dict[target_code]
                    format_prompt = f"""
                                    Please provide a single numerical value within the range {valid_scores}, along with a clear and concise explanation for your choice.\n\n
                                    Your response must be a dictionary in the following format:\n
                                    {{'score': <integer>, 'reason': <short explanation string>}}\n\n
                                    When writing the 'reason', please use a friendly and natural tone. Avoid overly formal or technical language.\n
                                    Strictly follow the output format. Do not include any extra text or modify the structure.\n
                                    """
                    score_list = []
                    reason_list = []
                    for fname in os.listdir(dir_path):
                        if fname.endswith(".json"):
                            continue
                        img_path = os.path.join(dir_path, fname)
                        full_prompt = f"{role_prompt}\n{task_prompt}\n{codebook[target_code]}\n{format_prompt}"
                        score, reason = self._run_for_score(full_prompt, img_path, valid_scores)
                        
                        score_list.append(score)
                        reason_list.append({fname:reason})

                    final_score = self._aggregate_scores(score_list, valid_scores)
                    image_score_dict[target_code] = [final_score,reason_list]

                results[f"{block_id}/{direction}"] = image_score_dict
                
        self.data_config.system_logger.info(f"Finalizing merge of annotations and saving results to {agent_annotation_path}.\n")
        self.generate_agent_anno_file(results,annotation_path,agent_annotation_path)
        return results