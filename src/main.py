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

    # m2 automated prompt tuner
    data_config.model_name = 'OpenGVLab/InternVL3-2B-hf'
    automated_prompt_tuner = AutomatedPromptTuner(data_config=data_config)
    
    # m3 vision langauge model processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vlm_processor = VLMProcessor(data_processor.data_config, device=device)

    # role prompt
    abstract_input_prompt = automated_prompt_tuner.construct_role_prompt()
    messages = automated_prompt_tuner.prepare_messages(abstract_input_prompt)
    role_prompt = automated_prompt_tuner.run(messages, vlm_processor)
    # print(role_prompt)
    # You are an expert in mixed methods research design and qualitative analysis, with a focus on understanding the shared and unique aspects of researcher versus adolescent observations in neighborhood environments. Your expertise includes using convergent mixed methods research to compare and contrast data sources and employing Key-Word-In-Context analysis to explore ethnic-racial label usage in ethnically/racially segregated neighborhoods.
        
    # m4 feedback provider





if __name__ == "__main__":
    main()