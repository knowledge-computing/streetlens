import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_HOME'] = '/data/vllm'

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
        print(f"Agent: Loading and annotating image from {image_path}...")
        while attempts < max_retries:
            try:            
                decoded_str = self.run(prompt, image_path=image_path, max_new_tokens=1024).strip()
                decoded_output = self.extract_dict_from_response(decoded_str)
                decoded_output = ast.literal_eval(decoded_output)
                print(f"Agent: My annotation is {decoded_output['score']} ")
                print(f"Agent: Generating explanation... Because {decoded_output['reason'][0].lower() + decoded_output['reason'][1:]}\n")
                if not isinstance(decoded_output, dict):
                    raise ValueError("Output is not a dictionary.")
                if 'score' not in decoded_output or 'reason' not in decoded_output:
                    raise KeyError("Missing 'score' or 'reason'")
                score = int(decoded_output['score'])
                if score not in valid_scores:
                    raise ValueError("Score not in valid range")
                return score
            except Exception as e:
                # print(f"Invalid response, retrying ({attempts + 1}/{max_retries}): {e}")
                attempts += 1
                
        # print("Maximum number of retries reached. Returning fallback score 99.")
        return 99

    def generate_agent_anno_file(self,output_dict,annotation_path,agent_annotation_path):
        import pandas as pd
        df = pd.read_csv(annotation_path)
        df.columns = df.columns.str.strip()
        new_rows = []
        for block_face, scores in output_dict.items():
            block_id, direction = block_face.split('/')
            row = {
                'Street Block ID': block_id.strip(),
                'Direction of Target Block Face': direction.strip()
            }
            row.update(scores)
            new_rows.append(row)
        df_agent = pd.DataFrame(new_rows)
        merged_df = pd.merge(df, df_agent, on=['Street Block ID', 'Direction of Target Block Face'], how='left', suffixes=('', '_agent'))
        merged_df.to_csv(agent_annotation_path, index=False)
        return merged_df

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
        for block_id in streetblock_ids:
            block_path = os.path.join(image_dir, block_id)
            for direction in os.listdir(block_path):
                dir_path = os.path.join(block_path, direction)
                image_score_dict = {}

                for target_code in target_codes:
                    print(f"Agent: Jumping into code theme {target_code}...\n")
                    valid_scores = class_dict[target_code]
                    format_prompt = f"""
                                    Please provide a single numerical value within the range {valid_scores}, along with a clear and concise explanation for your choice.\n\n
                                    Your response must be a dictionary in the following format:\n
                                    {{'score': <integer>, 'reason': <short explanation string>}}\n\n
                                    Strictly follow the output format. Do not include any extra text or modify the structure.
                                    """

                    score_list = []
                    for fname in os.listdir(dir_path):
                        if fname.endswith(".json"):
                            continue
                        img_path = os.path.join(dir_path, fname)
                        full_prompt = f"{role_prompt}\n{codebook[target_code]}\n{format_prompt}"
                        score = self._run_for_score(full_prompt, img_path, valid_scores)
                        
                        score_list.append(score)

                    final_score = self._aggregate_scores(score_list, valid_scores)
                    image_score_dict[target_code] = final_score

                results[f"{block_id}/{direction}"] = image_score_dict

        print (f"\nAgent: Merging my annotations and saving output to {agent_annotation_path} ... \n=============")
        self.generate_agent_anno_file(results,annotation_path,agent_annotation_path)
        return results