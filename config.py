from pathlib import Path

def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 3,
        "lr": 1e-3,
        "seq_len": 128,
        "d_model": 512,
        "limit_train_instances": 10**5, # limit of data points for training
        "limit_valid_instances": 500,   # limit of data points for validation
        "chunk_size": 50,              # size of every indpendent sequence in words
        "dataset_name": "allenai/peS2o",
        "model_folder": "weights",
        "model_basename": "lmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer.json",
    }

def get_weights_file_path(config, epoch: str):
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / config['model_folder'] / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(config['model_folder']).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])