import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Transformer(nn.Module):
  """A transformer model"""

  def __init__(self, vocab_size) -> None:
    super(Transformer, self).__init__()

    h = 8               # number of attention heads within each layer
    d_model = 512       # model dimension
    d_k = d_model // h  # dimension of query and key tensors
    n = 5               # number of encoder / decoder layers
    max_len = 512       # maximum number of tokens

    # initialize token and positional embeddings
    self.tok_emb = nn.Embedding(vocab_size, d_model)
    self.pos_emb = PositionalEncoding(d_model, max_len)

    # initialize `n` transformer layers
    self.layers = nn.ModuleList([TransformerLayer(d_model, d_k, h, max_len) for _ in range(n)])

    # initialize the prediction layer
    self.pred_layer = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    """
    Forward pass of the Transformer Model
    :param x: tensor of shape (batch_size x chunk_len)
    :return y_hat: prediction tensor of shape (batch_size x chunk_len x vocab_size)
    """

    # apply token and position embedding
    x = self.tok_emb(x) + self.pos_emb(x)

    # apply transformer layers in sequence
    for layer in self.layers:
      x = layer(x)

    # apply prediction layer
    return self.pred_layer(x)
    

class PositionalEncoding(nn.Module):
  """
  Positional encoding
  Add positional information to the token embedding.
  Without it the model wouldn't be able to differentiate between
  different character orders such as between "dogs" and "god"
  """

  def __init__(self, d_model, max_len=1000):
    super().__init__()

    pos = torch.arange(max_len).unsqueeze(1).float() # positions in a sequence
    scale = 1000.0 ** (torch.arange(0, d_model, 2).float() / d_model)

    PE = torch.zeros(max_len, d_model)
    PE[:, 0::2] = torch.sin(pos/scale)  # set even columns to sin(pos/scale)
    PE[:, 1::2] = torch.cos(pos/scale)  # set odd columns to cos(pos/scale)
    self.PE = PE.unsqueeze(0).cuda()
    self.PE.requires_grad = False

  def forward(self, x):
    """
    Compute positional encoding for x
    :param x: input tensor of shape (batch_size x chunk_len)
    :return x_hat: output tensot of shape (batch_size x chunk_len x d_model)
    (i.e obtain a pos. embedding vector for each of chunk_len tokens)
    """
    # return a slice (with rows up to chunk_len) of the pre-computed position encoding
    chunk_len = x.size(dim=1)
    return self.PE[:, :chunk_len]


class TransformerLayer(nn.Module):
  """A single transformer layer."""
  
  def __init__(self, d_model, d_k, h, max_len) -> None:
    super().__init__()
    
    # initialize multi head attention block
    self.multi_head_attn = MultiHeadAttention(d_model, d_k, h, max_len)

    # initialize layer normalization blocks
    self.l_norm1 = torch.nn.LayerNorm(d_model)
    self.l_norm2 = torch.nn.LayerNorm(d_model)

    # initialize feed forward network
    self.ffn = PositionFeedForwardNet(d_model)

  def forward(self, x):
    """
    Forward pass of a single transformer layer
    :param x: tensor of shape (batch_size x chunk_len)
    :return y_hat: prediction tensor of shape (batch_size x chunk_len x d_model)
    """
    mha = self.multi_head_attn(x)
    out = self.l_norm1(mha + x)
    return self.l_norm2(self.ffn(out) + out)
  

class MultiHeadAttention(nn.Module):
  """
  Multi Head Attention
  Note that the implementation is optimized such that
  we can perform `h` (number of heads) scaled dot product self
  attentions at in parallel.
  """

  def __init__(self, d_model, d_k, h, max_len):
    super().__init__()
    self.h = h
    self.d_k = d_k

    # initialize linear projections layers
    self.w_q = nn.Linear(d_model, h * d_k)
    self.w_k = nn.Linear(d_model, h * d_k)
    self.w_v = nn.Linear(d_model, h * d_k)

    # initialize softmax and mask
    self.softmax = nn.Softmax(dim=-1)
    self.mask_opt = AttentionMasking(max_len)

    # initialize the output projection
    self.w_o = nn.Linear(h * d_k, d_model)

  def forward(self, x):
    """
    Forward method of Multi Head Attention layer
    :param x: input tensor of shape (batch_size x chunk_len x d_model)
    :param y_hat: output tensor of shape (batch_size x chunk_len x d_model)
    """
    
    # apply projections and split q, k, v such that we can compute
    # `h` scaled dot product attention operations in parallel
    q = self.head_split(self.w_q(x))
    k = self.head_split(self.w_k(x))
    v = self.head_split(self.w_v(x))

    _, _, _, d_k = q.shape
    out = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    out = self.softmax(self.mask_opt(out))
    out = out @ v

    # concatenate attention outputs from the `h`
    # scaled dot product attention heads
    out = self.head_concat(out)
    return self.w_o(out)
  

  def head_split(self, x):
    """
    Split tensor between attention heads
    :param x: input tensor of shape (batch_size x chunk_length x d_model)
    :return x_split: output tensor of shape (batch_size x h x chunk_length x d_k)
    """
    batch_size, chunk_len, _ = x.shape
    return x.view(batch_size, chunk_len, self.h, self.d_k).transpose(1, 2)
  
  def head_concat(self, x_split):
    """
    Concatenate head tensor
    :param x_split: input tensor of shape (batch_size x h x chunk_length x d_k)
    :param x: output tensor of shape (batch_size x chunk_length x d_model)
    """
    batch_size, h, chunk_length, d_k = x_split.shape
    return x_split.transpose(1, 2).contiguous().view(
      batch_size, chunk_length, h * d_k)


class PositionFeedForwardNet(nn.Module):
  """Simple 2 layer feed forward network"""

  def __init__(self, d_model) -> None:
    super().__init__()

    self.linear1 = nn.Linear(d_model, 4*d_model)
    self.linear2 = nn.Linear(4*d_model, d_model)

  def forward(self, x):
    return self.linear2(torch.nn.functional.relu(self.linear1(x)))


class AttentionMasking(nn.Module):
  """Attention masking block"""

  def __init__(self, max_len) -> None:
    super().__init__()
    # compute the lower triangular matrix
    tril = torch.tril(torch.ones(max_len, max_len))
    tril = tril.view((1, 1, max_len, max_len))
    self.register_buffer("mask", tril)
    
  def forward(self, attn_weights):
    """
    :param attn_weights: input tensor of shape (batch_size, h, chunk_len, chunk_len)
    :param attn_weights_masked: output tensor of the same shape, where weights
    corresponding to tokens ("in the future") are masked out.  
    """
    chunk_len = attn_weights.shape[-1]
    # select a subset of the precomputed mask and set location where mask is zero to -inf
    attn_weights_masked = attn_weights.masked_fill(self.mask[:, :, :chunk_len, :chunk_len] == 0, float('-inf'))
    return attn_weights_masked


def test_shapes():
  vocab_size = 100 # vocabulary size (number of unique chars)
  batch_size = 1  # number of sequences in a batch
  chunk_len = 128  # number of characters in each sequence

  x = torch.randint(0, vocab_size, (batch_size, chunk_len)).cuda()
  y = torch.randint(0, vocab_size, (batch_size, chunk_len)).cuda()
  print(x.shape, y.shape)

  model = Transformer(vocab_size).cuda()
  y_hat = model(x) # (batch_size x chunk_len x vocab_size)
  assert y_hat.shape[:-1] == y.shape, \
    f"Output shape is incorrect ! y_hat.shape: {y_hat.shape}, y.shape: {y.shape}"
  print("test_shapes passed :)")


if __name__ == "__main__":
  test_shapes()
