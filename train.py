
from model import build_transformer
from dataset import Dataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def get_all_sentences(ds):
    for item in ds:
        yield item

def get_or_build_tokenizer(config, ds):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('json', data_files=config["dataset_path"], split='train')

    tokenizer = get_or_build_tokenizer(config, ds_raw)

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = Dataset(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = Dataset(val_ds_raw, tokenizer, config['seq_len'])
   
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer

def get_model(config, vocab_size): 
    model = build_transformer(vocab_size, config["seq_len"], config['d_model'])
    return model

def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    
    # Make sure the weights folder exists
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer = get_ds(config) # get data and tokenizers
    model = get_model(config, tokenizer.get_vocab_size()).to(device) # get model ?

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()    
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:

            decoder_input = batch['decoder_input'].to(device) 
            mask = batch['mask'].to(device)

            decoder_output = model.decode(decoder_input, mask)
            proj_output = model.project(decoder_output) 

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1)) 

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, lambda msg: batch_iterator.write(msg))

        # Save the model after each epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")

        print("Saving model to:", model_filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_trgt, max_len, device, print_msg, num_examples=2):
    # here
    model.eval()
    count = 0

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # if we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds: # for every batch
            count += 1
            encoder_input = batch["encoder_input"].to(device) # encoder input for a batch=1, 1 tokenied sentence
            encoder_mask = batch["encoder_mask"].to(device) # mask fir the encoder

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_trgt, max_len, device) # prediction of the sentence in tokens

            source_text = batch["src_text"][0] # original source sentence
            target_text = batch["trgt_text"][0] # original target sentence
            model_out_text = tokenizer_trgt.decode(model_out.detach().cpu().numpy()) # prediction of the model about target sentence
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples: # how many examples to show during validation (*by eye*, metrics could be added)
                print_msg('-'*console_width)
                break

