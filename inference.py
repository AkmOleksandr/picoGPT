from pathlib import Path
from config import latest_weights_file_path 
from model import build_transformer
from tokenizers import Tokenizer
import torch
import torch.nn.functional as F

@torch.no_grad()
def get_response(config, text, temperature=0.8, top_k=None):
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
        ).unsqueeze(0)
    
    final_output = ""
    # Generate output word by word
    while decoder_input.size(1) < config['seq_len']:

        decoder_output = model.decode(decoder_input)

        probs = model.project(decoder_output[:, -1]) # get probabilities for the next token
        
        next_token = _get_next_token(probs, temperature, top_k)

        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).long().fill_(next_token).to(device)], dim=1)

        next_word = tokenizer.decode([next_token])

        # Break if <EOS> was predicted
        if next_word == "[EOS]":
            break

        final_output += next_word + ' '

    return final_output.strip()

def _get_next_token(probs, temperature, top_k):
    if temperature == 0:
        _, next_token = torch.max(probs, dim=1).item()  # select token with the highest probability
        return next_token
    elif temperature > 0:
        scaled_probs = F.softmax(probs / temperature, dim=1)
        
        if top_k is not None:
            # Apply top-k filtering
            values, indices = torch.topk(scaled_probs, top_k, dim=1)
            scaled_probs = torch.zeros_like(scaled_probs).scatter(1, indices, values)
        
        next_token = torch.multinomial(scaled_probs, 1).item()
        return next_token
    else:
        raise ValueError("Temperature must be a positive value.")
