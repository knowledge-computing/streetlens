class DataProcessor:
    class Config:
        def __init__(self, codebook_path: str, annotation_path: str, paper_path: str, street_block_id: list, protocol_path: str=None):
        self.codebook_path = codebook_path
        self.annotation_path = annotation_path
        self.paper_path = paper_path

        self.street_block_id = street_block_id
        self.protocol_path = protocol_path
    
    def __init__(self, config):
        self.config = config

    def _set_model_name(self, model_name):
        self.config.model_name = model_name