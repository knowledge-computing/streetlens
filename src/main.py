from data_processor import *
from automated_prompt_tuner import *
from vision_language_model_processor import *
from feedback_provider import *

def main():
    # todo get input for path
    codebook_path = './dataset/annotation/sso_codebook.json'
    paper_path = './dataset/paper/abstract.json'
    annotation_path = './dataset/annotation/sso_annotation.csv'
    street_block_id = ['62146', '281', '282', '9576']
    # protocol_path

    config = Config(codebook_path=codebook_path, annotation_path=annotation_path, paper_path=paper_path, street_block_id=street_block_id)

    # data processor

    # automated prompt tuner
    config._set_model_name = 'OpenGVLab/InternVL3-2B-hf'

    # vision langauge model processor

    # feedback provider

if __name__ == "__main__":
    main()