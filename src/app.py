import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_HOME'] = '/data/vllm'

from flask import Flask, request, jsonify
from flask_cors import CORS
from io import StringIO
import logging
import torch

from data_processor import DataProcessor
from automated_prompt_tuner import AutomatedPromptTuner
from vision_language_model_processor import VLMProcessor


app = Flask(__name__)
CORS(app) 

def create_logger(name):
    log_stream = StringIO()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    logger.handlers = []

    handler = logging.StreamHandler(log_stream)
    formatter = logging.Formatter('[%(name)s] %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger, log_stream

codebook_path = './dataset/annotation/sso_codebook.json'
paper_path = './dataset/paper/abstract.json'
annotation_path = './dataset/annotation/sso_annotation.csv'
agent_annotation_path = './dataset/annotation/agent_annotation.csv'
street_block_id = ['281']

device = "cuda" if torch.cuda.is_available() else "cpu"

data_config = DataProcessor.DataConfig(
    codebook_path=codebook_path,
    annotation_path=annotation_path,
    paper_path=paper_path,
    street_block_id=street_block_id
)
data_config.model_name = 'OpenGVLab/InternVL3-2B-hf'

data_processor = DataProcessor(data_config=data_config)
vlm_processor = VLMProcessor(data_config=data_config, device=device)
automated_prompt_tuner = AutomatedPromptTuner(data_config=data_config, vlm_processor=vlm_processor)

@app.route("/api/data_processor", methods=["POST"])
def run_data_processor():
    system_logger, log_stream = create_logger("StreetLens")
    data_config.system_logger = system_logger
    return jsonify({"logs": log_stream.getvalue()})

@app.route("/api/prompt_tuner", methods=["POST"])
def run_prompt_tuner():
    system_logger, system_stream = create_logger("StreetLens")
    agent_logger, agent_stream = create_logger("Stella")
    data_config.system_logger = system_logger
    data_config.agent_logger = agent_logger

    automated_prompt_tuner.construct_role_prompt()
    automated_prompt_tuner.identify_task_type()
    automated_prompt_tuner.construct_codebook_prompt()

    logs = f"{system_stream.getvalue()}\n{agent_stream.getvalue()}"
    return jsonify({"logs": logs})

@app.route("/api/vlm_processor", methods=["POST"])
def run_vlm_processor():
    system_logger, system_stream = create_logger("StreetLens")
    agent_logger, agent_stream = create_logger("Stella")
    data_config.system_logger = system_logger
    data_config.agent_logger = agent_logger

    data_config.role_prompt = automated_prompt_tuner.role_prompt
    data_config.task_types_dict = automated_prompt_tuner.task_types_dict
    data_config.codebook_prompt_dict = automated_prompt_tuner.codebook_prompt_dict
    data_config.agent_annotation_path = agent_annotation_path
    data_config.image_dir = './dataset/img/'

    vlm_processor.generate_annotation()

    logs = f"{system_stream.getvalue()}\n{agent_stream.getvalue()}"
    return jsonify({"logs": logs})
    

if __name__ == "__main__":
    app.run(debug=True)
