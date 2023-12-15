from model import build_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import LLMDataset, get_sequences

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_or_build_tokenizer(config, dataset):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        
        sequences = get_sequences(dataset, "train", config['limit_train_instances'], config['chunk_size'])
        tokenizer.train_from_iterator(sequences, trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_data(config):
    train_dataset = load_dataset(config['dataset_name'], split="train", streaming=True)
    valid_dataset = load_dataset(config['dataset_name'], split="validation", streaming=True)

    tokenizer = get_or_build_tokenizer(train_dataset)

    # Extract independent sequences given limitations
    train_sequences = get_sequences(train_dataset, "train", config['limit_train_instances'], config['chunk_size'])
    valid_sequences = get_sequences(valid_dataset, "valid", config['limit_valid_instances'], config['chunk_size'])

    train_ds_obj = LLMDataset(train_sequences, tokenizer, config['seq_len'])
    valid_ds_obj = LLMDataset(valid_sequences, tokenizer, config['seq_len'])
   
    train_dataloader = DataLoader(train_ds_obj, batch_size=config['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_ds_obj, batch_size=1, shuffle=True)

    return train_dataloader, valid_dataloader, tokenizer

def get_model(config, vocab_size): 
    model = build_transformer(vocab_size, config['seq_len'], config['d_model'])
    return model

def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    
    # Make sure the weights folder exists
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, valid_dataloader, tokenizer = get_data(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == "latest" else get_weights_file_path(config, preload) if preload else None

    if model_filename: # Load model if one exists
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()    
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:

            decoder_input = batch["decoder_input"].to(device)

            decoder_output = model.decode(decoder_input)
            proj_output = model.project(decoder_output) 

            label = batch["label"].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            loss.backward() # teaching model to predict next word at every position by minimizing the loss between data distribution across vocab_size of current token with the actual one-hot encoded label of the next token

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        run_validation(model, valid_dataloader, tokenizer, config['seq_len'], device, lambda msg: batch_iterator.write(msg))

        # Save the model after each epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")

        print("Saving model to:", model_filename)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)

@ torch.no_grad()
def run_validation(model, validation_ds, tokenizer, seq_len, device, print_msg, num_examples=5):
    model.eval()
    count = 0
    losses = []
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    # Get the console window width
    try:
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    for batch in validation_ds:
        count += 1
        decoder_input = batch['decoder_input'].to(device) 

        decoder_output = model.decode(decoder_input)
        proj_output = model.project(decoder_output) # (batch_size, seq_len, vocab_size) every batch is a matrix with probability distribution of every word being at that position as a row and the vocabulary size as the number of columns

        label = batch["label"].to(device) # (batch_size, seq_len) every batch is a vector of tokens for every position in seq_len

        loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
        losses.append(loss)

        model_out = greedy_decode(model, tokenizer, seq_len, device) # prediction of the sentence in tokens

        model_out_text = tokenizer.decode(model_out.detach().cpu().numpy()) # prediction of the model in words
        
        print_msg("-"*console_width)
        print_msg(f"{f'Model says: ':>12}{model_out_text}")

        if count == num_examples:
            print_msg("-"*console_width)
            break

    print_msg("Average validation loss:", torch.mean(losses))
    print_msg("Standard deviation of the validation loss:", torch.std(losses))

def greedy_decode(model, tokenizer, seq_len, device):

    sos_idx = tokenizer.token_to_id("[SOS]") # get <SOS> token
    eos_idx = tokenizer.token_to_id("[EOS]") # get <EOS> token

    decoder_input = torch.empty(1, 1).fill_(sos_idx).to(device) # initialize the decoder input with <SOS>

    while decoder_input.size(1) < seq_len:
    
        out = model.decode(decoder_input)

        probs = model.project(out[:, -1]) # probabilities of the next token
        
        _, next_word = torch.max(probs, dim=1) # get the token with max prob

        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).fill_(next_word.item()).to(device)], dim=1) # append next_word (the predicted word) to decoder_input

        if next_word == eos_idx: # if next token is <EOS> break
            break

    return decoder_input.squeeze(0)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)