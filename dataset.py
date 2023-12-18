from itertools import islice
import re
import torch
from torch.utils.data import Dataset

class LLMDataset(Dataset):

    def __init__(self, dataset, tokenizer, seq_len): 
        super().__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Create tokens for special words
        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64) 
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError("Index out of range")
        # Get item from dataset with index idx
        text = self.dataset[idx]
        tokenized_text = self.tokenizer.encode(text).ids 
        
        if len(tokenized_text) + 1 > self.seq_len:
          tokenized_text = tokenized_text[:self.seq_len-1] # handling edge case when tokenized_text exceeds seq_len 
        # How much padding we need to add to reach seq_len 
        num_padding_tokens = self.seq_len - len(tokenized_text) - 1 # -1 because <SOS> in decoder_input and <EOS> in label
        # Concat: <SOS>, tokenized text, padding
        decoder_input = torch.cat( # start at <SOS>, so we train the model to predict next token at every position because labels start at text
            [
                self.sos_token,
                torch.tensor(tokenized_text, dtype=torch.int64),
                torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Concat: tokenized text, <EOS>, padding
        label = torch.cat(
            [
                torch.tensor(tokenized_text, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure both are seq_len long
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "decoder_input": decoder_input,  # (seq_len) tokenized sentence in time t
            "label": label,                  # (seq_len) tokenized sentence in time t+1
            "text": text,
        }
    
# Handle data due to computational and memory constraints

def get_sequences(dataset, split, limit_instances, chunk_size): # get a list of tokenization-ready independent sequenecs
    all_chunks = []
    original_sequences = _get_original_sequences(dataset, split, limit_instances)
    # Iterate over original sequences and split them into chunks
    for sequence in original_sequences:
        chunks = _create_text_chunks(sequence, chunk_size)
        all_chunks.extend(chunks)
    return all_chunks

def _get_original_sequences(dataset, split, limit_instances):
    if split != "train" and split != "valid":
        raise ValueError("Invalid split value. Use 'train' or 'valid'.")
    return [item["text"] for item in islice(dataset, limit_instances) if item["source"] == f"s2ag/{split}"]

def _create_text_chunks(text, chunk_size):
    # Split the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    chunks = []
    current_chunk = ""
    current_chunk_size = 0

    for sentence in sentences:
        # Check if adding the sentence to the current chunk exceeds the chunk size
        sentence_length = len(sentence.split())
        if current_chunk_size + sentence_length + 1 <= chunk_size:
            if current_chunk:
                current_chunk += " "  # Add space between sentences
            current_chunk += sentence
            current_chunk_size += sentence_length
        else:
            # Add the current chunk to the list and start a new chunk
            chunks.append(current_chunk)
            current_chunk = sentence
            current_chunk_size = sentence_length

    # Add the last chunk to the list
    if current_chunk:
        chunks.append(current_chunk)
    return chunks