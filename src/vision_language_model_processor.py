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
        import torchvision.transforms as T
        image = Image.open(path).convert("RGB")
        return T.ToTensor()(image)

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

