import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_HOME'] = '/data/vllm'

from data_processor import *
from automated_prompt_tuner import *
from vision_language_model_processor import *
from feedback_provider import *

def main():
    # todo: get input for path
    codebook_path = './dataset/annotation/sso_codebook.json'
    paper_path = './dataset/paper/abstract.json'
    annotation_path = './dataset/annotation/sso_annotation.csv'
    street_block_id = ['62146', '281', '282', '9576']
    # protocol_path

    # m1 data processor
    data_config = DataProcessor.DataConfig(codebook_path=codebook_path, annotation_path=annotation_path, paper_path=paper_path, street_block_id=street_block_id)
    data_processor = DataProcessor(data_config=data_config)

    # m2 automated prompt tuner & m3 vision langauge model processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vlmprocessor = VLMProcessor(data_processor.data_config, model_name='OpenGVLab/InternVL3-2B-hf', device=device)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/throughput_smolvlm.png"},
                 {"type": "text", "text": "What is this chart about?"},
                ],
            },
    ]

    inputs = vlmprocessor.processor.apply_chat_template(
        messages,
        padding=True,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(vlmprocessor.model.device, dtype=torch.bfloat16)

    outputs = vlmprocessor.model.generate(**inputs, max_new_tokens=25)
    decoded_outputs = vlmprocessor.processor.batch_decode(outputs, skip_special_tokens=True)
    print(decoded_outputs)
        
    # m4 feedback provider





if __name__ == "__main__":
    main()