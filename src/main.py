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

    # m1 data processor
    data_config = DataProcessor.DataConfig(codebook_path=codebook_path, annotation_path=annotation_path, paper_path=paper_path, street_block_id=street_block_id)
    data_processor = DataProcessor(data_config=data_config)

    # m3 vision langauge model processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_config.model_name = 'OpenGVLab/InternVL3-2B-hf'
    vlm_processor = VLMProcessor(data_processor.data_config, device=device)

    # m2 automated prompt tuner
    automated_prompt_tuner = AutomatedPromptTuner(data_config=data_config, vlm_processor=vlm_processor)
    
    # role prompt
    automated_prompt_tuner.construct_role_prompt()

    # m4 feedback provider





if __name__ == "__main__":
    main()