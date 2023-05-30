import numpy as np
import mindspore
from mindspore import Tensor

from .cpm_bee import CpmBeeRotaryEmbedding, CpmBeeEmbeddingExt, CpmDenseGatedACT

def test_cpm_bee_rotary_embedding():
    """
        Args:
        x (:obj:`Tensor` of shape ``(..., dim)``): Inputs.
        x_pos (:obj:`Tensor` of shape ``(...)``): Positions of inputs.
    """
    vocab_size = 1000
    batch_size = 2
    seq_length = 16
    hidden_size = 32
    x = Tensor(np.random.randn(batch_size, seq_length, hidden_size), mindspore.float32)
    x_pos = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)), mindspore.int32)

    embedding = CpmBeeRotaryEmbedding(hidden_size)

    output = embedding(x, x_pos)

    assert output.shape == (batch_size, seq_length, hidden_size)

def test_cpm_bee_embedding_ext():
    """
        Args:
            ids (:obj:`Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
            ids (:obj:`Tensor` of shape ``(batch_size)``): Subscript of input sequence tokens.
        Return:
            :obj:`Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
    """
    vocab_size = 1000
    batch_size = 2
    seq_length = 16
    hidden_size = 32
    ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)), mindspore.int32)
    ids_sub = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_length)), mindspore.int32)

    embedding = CpmBeeEmbeddingExt(vocab_size, hidden_size)

    output = embedding(ids, ids_sub)

    assert output.shape == (batch_size, seq_length, hidden_size)

def test_cpm_dense_gated_act():
    """
    Args:
        hidden_states (`Tensor` of shape `(batch, seq_len, dim_in)`)
    """
    batch_size = 2
    seq_length = 16
    hidden_size = 256
    dim_ff = hidden_size * 4
    dense_gated_act = CpmDenseGatedACT(hidden_size, dim_ff)

    hidden_states = Tensor(np.random.randn(batch_size, seq_length, hidden_size), mindspore.float32)

    output = dense_gated_act(hidden_states)

    assert output.shape == (batch_size, seq_length, dim_ff)
