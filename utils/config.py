from pathlib import Path

class ModelConfig:
    def __init__(self):
        self.batch_size = 8
        self.num_epochs = 20
        self.lr = 10**-4
        self.seq_len = 350
        self.d_model = 512
        self.datasource = 'opus_books'
        self.lang_src = "en"
        self.lang_tgt = "it"
        self.model_folder = "weights"
        self.model_basename = "tmodel_"
        self.preload = ""  # Empty string instead of None
        self.tokenizer_file = "tokenizer_{0}.json"
        self.experiment_name = "runs/tmodel"
        self.wandb_group = "exp1"  # Added this field
        # These will be populated by train.py
        self.local_rank = -1
        self.global_rank = -1

def get_default_config():
    return ModelConfig()

def get_weights_file_path(config, epoch):
    model_folder = f"{config.datasource}_{config.model_folder}"
    model_filename = f"{config.model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def get_latest_weights_file_path(config):
    model_folder = f"{config.datasource}_{config.model_folder}"
    model_filename = f"{config.model_basename}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])