import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # initialize dimensions of embedding with number of unique units in vocabulary and size of embedding (size of vector to represent every word)

    def forward(self, X):
        return self.embedding(X) * math.sqrt(self.d_model) # apply it on input data and multiply by sqrt of embedding size 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        PE = torch.zeros(seq_len, d_model) # every row will be an embedding of a single unit is in which units came in sentence
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)
        PE = PE.unsqueeze(0)
        self.register_buffer("PE", PE)

    def forward(self, X):
        X = X + (self.PE[:, :X.shape[1], :]).requires_grad_(False) # embedding matrix as input, append position of every word in the sentence (same for every sentence)
        return self.dropout(X)

class LayerNormalization(nn.Module): # normalize 
    def __init__(self, num_features, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, X):

        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)

        return self.alpha * (X - mean) / (std + self.eps) + self.bias # return normalized matrix

class ResidualConnection(nn.Module): # connecter
    def __init__(self, num_features, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(num_features)

    def forward(self, X, sublayer):
        return X + self.dropout(sublayer(self.norm(X))) # combines embedding matrix with normalized embedding matrix
    
class FeedForwardBlock(nn.Module): # copletely optional architecture
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, X):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(X))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # size of every head

        # feed forward layers
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def attention(Q, K, V, mask, dropout):

        d_k = Q.shape[-1]

        attention_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k) # formula for attention

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # masked_fill_ method in PyTorch that replaces elements in the input tensor (attention_scores in this case) with a specified value where the corresponding element in the mask tensor is 0 (mask either hides padding or padding and next words)
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ V)
    
class DecoderBlock(nn.Module):
    def __init__(self, num_features, self_attention_block, cross_attention_block, feed_forward_block, dropout):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(num_features, dropout) for _ in range(3)])

    def forward(self, X, mask):
        # here
        X = self.residual_connections[0](X, lambda X: self.self_attention_block(X, X, X, mask)) # receives decoder input (trgt lang in this case as input)
        X = self.residual_connections[2](X, self.feed_forward_block)
        
        return X

class Decoder(nn.Module):
    def __init__(self, num_features, layers):
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(num_features)

    def forward(self, X, encoder_output, src_mask, trgt_mask):

        for layer in self.layers:
            X = layer(X, encoder_output, src_mask, trgt_mask) # pass through all layers (N Encoder Blocks)

        return self.norm(X)
    
class ProjectionLayer(nn.Module): # maps embedding with positions in vocabulary
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, X):
        
        return self.proj(X) # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
    

class Transformer(nn.Module):
    def __init__(self, decoder: Decoder, src_embed: InputEmbeddings, trgt_embed: InputEmbeddings, src_pos: PositionalEncoding, trgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.decoder = decoder
        self.src_embed = src_embed
        self.trgt_embed = trgt_embed
        self.src_pos = src_pos
        self.trgt_pos = trgt_pos
        self.projection_layer = projection_layer

    # Every sentence is a single batch, which is a matrix of seq_len (max length of sentence) x d_model (embedding size)

    
    def decode(self, encoder_output, src_mask, trgt, trgt_mask): # replace encoder_output and remove trgt/src
        # (batch, seq_len, d_model)
        trgt = self.trgt_embed(trgt)
        trgt = self.trgt_pos(trgt)
        return self.decoder(trgt, encoder_output, src_mask, trgt_mask)
    
    def project(self, X):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(X)