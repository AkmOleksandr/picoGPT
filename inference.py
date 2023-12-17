from pathlib import Path
from config import latest_weights_file_path 
from model import build_transformer
from tokenizers import Tokenizer
import torch

@torch.no_grad()
def get_response(config, text):
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_file'])))
    model = build_transformer(tokenizer.get_vocab_size(), config['seq_len'], config['d_model']).to(device)
    
    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    model.eval()
    
    tokenizer_path = Path(config['tokenizer_file'])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenized_text = tokenizer.encode(text).ids

    sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64).to(device)
    decoder_input = torch.cat( 
            [
                sos_token,
                torch.tensor(tokenized_text, dtype=torch.int64).to(device),
            ],
            dim=0,
        )
    
    final_output = ""
    # Generate output word by word
    print(decoder_input.size())
    while decoder_input.size(1) < config['seq_len']:

        decoder_output = model.decode(decoder_input)

        probs = model.project(decoder_output[:, -1]) # get probabilities for the next token
        _, next_token = torch.max(probs, dim=1) # select one with the highest probability

        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).long().fill_(next_token.item()).to(device)], dim=1)

        next_word = tokenizer.decode([next_token.item()])

        # Break if <EOS> was predicted
        if next_word == "[EOS]":
            break

        final_output += next_word + ' '

    return final_output.strip()