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
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, X):
        return self.net(X)

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
        attention_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask:
            size = attention_scores.size(-1) # extract dimension of features
            lower_triangular_matrix = torch.tril(torch.ones((1, size, size))).type(torch.int)
            attention_scores.masked_fill_(lower_triangular_matrix == 0, -1e9) # fills upper triangle with -1e9 (to make them 0s after softmax) 
        
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ V)
    
class DecoderBlock(nn.Module):
    def __init__(self, num_features, multi_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.multi_attention_block = multi_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(num_features, dropout) for _ in range(2)])

    def forward(self, X):
        X = self.residual_connections[0](X, lambda X: self.multi_attention_block(X, X, X, mask=True, dropout=0.1))
        X = self.residual_connections[1](X, self.feed_forward_block)
        return X

class Decoder(nn.Module):
    def __init__(self, num_features, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(num_features)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return self.norm(X)
    
class ProjectionLayer(nn.Module): # maps embedding with positions in vocabulary
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, X):
        return self.proj(X) # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
    
class Transformer(nn.Module):
    def __init__(self, decoder: Decoder, embed: InputEmbeddings, pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.decoder = decoder
        self.embed = embed
        self.pos = pos
        self.projection_layer = projection_layer

    # Every sentence is a single batch, which is a matrix of seq_len (max length of sentence) x d_model (embedding size)
    def decode(self, X):
        # (batch, seq_len, d_model)
        embeds = self.embed(X)
        decoder_input = self.pos(embeds)
        return self.decoder(decoder_input)
    
    def project(self, X):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(X)
    
def build_transformer(vocab_size, seq_len, d_model=512, N=6, h=8, dropout=0.15) -> Transformer:

    embed = InputEmbeddings(d_model, vocab_size)
    pos = PositionalEncoding(d_model, seq_len, dropout)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, vocab_size)
    
    transformer = Transformer(decoder, embed, pos, projection_layer)
    
    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer