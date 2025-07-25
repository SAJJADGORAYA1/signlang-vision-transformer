import torch
import math
from torch import nn

# Config (copied here for model dependency)
MAX_FRAMES = 30
MODEL_DIM = 512
NUM_HEADS = 4
NUM_LAYERS = 2
KEYPOINT_DIM = (33 * 4) + (21 * 3) + (21 * 3) + (468 * 3)  # 1662

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_k, d_v, d_model):
        super().__init__()
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        
        self.W_q = nn.Linear(d_model, h * d_k)
        self.W_k = nn.Linear(d_model, h * d_k)
        self.W_v = nn.Linear(d_model, h * d_v)
        self.W_o = nn.Linear(h * d_v, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.h, self.d_v).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_v)
        return self.W_o(output)

class AddNormalization(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, sublayer_output):
        return self.layer_norm(x + sublayer_output)

class FeedForward(nn.Module):
    def __init__(self, d_ff, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class DecoderLayer(nn.Module):
    def __init__(self, h, d_k, d_v, d_model, d_ff, dropout_rate):
        super().__init__()
        self.multihead_attention1 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.add_norm1 = AddNormalization(d_model)
        
        self.multihead_attention2 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.add_norm2 = AddNormalization(d_model)
        
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.add_norm3 = AddNormalization(d_model)
    
    def forward(self, x, encoder_output, lookahead_mask=None, padding_mask=None):
        attn1 = self.multihead_attention1(x, x, x, lookahead_mask)
        attn1 = self.dropout1(attn1)
        addnorm1 = self.add_norm1(x, attn1)
        
        attn2 = self.multihead_attention2(addnorm1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2)
        addnorm2 = self.add_norm2(addnorm1, attn2)
        
        ff_output = self.feed_forward(addnorm2)
        ff_output = self.dropout3(ff_output)
        return self.add_norm3(addnorm2, ff_output)

class Decoder(nn.Module):
    def __init__(self, d_model, max_seq_len, h, d_k, d_v, d_ff, n_layers, dropout_rate):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(h, d_k, d_v, d_model, d_ff, dropout_rate) 
            for _ in range(n_layers)
        ])
    
    def forward(self, x, encoder_output, lookahead_mask=None, padding_mask=None):
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, lookahead_mask, padding_mask)
        return x

class SignTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes, 
                 h=4, d_k=128, d_v=128, d_ff=2048, 
                 n_layers=2, dropout_rate=0.1, max_seq_len=MAX_FRAMES):
        super().__init__()
        self.src_proj = nn.Linear(input_dim, model_dim)
        self.dummy_encoder = nn.Parameter(torch.randn(1, MAX_FRAMES, model_dim))
        self.decoder = Decoder(
            d_model=model_dim,
            max_seq_len=max_seq_len,
            h=h,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout_rate=dropout_rate
        )
        self.classifier = nn.Linear(model_dim, num_classes)
    
    def forward(self, src):
        x = self.src_proj(src)
        encoder_output = self.dummy_encoder.repeat(x.size(0), 1, 1)
        decoder_output = self.decoder(x, encoder_output)
        pooled = decoder_output.mean(dim=1)
        return self.classifier(pooled)