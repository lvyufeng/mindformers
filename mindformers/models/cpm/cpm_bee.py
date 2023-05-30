# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""GPT model"""
import copy
import numpy as np

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindformers.modules.transformer.moe import default_moe_config
from mindformers.modules.layers import CpmLayerNorm, Dropout
from mindformers.core.loss import CrossEntropyLoss
from mindformers.modules.transformer.transformer import CpmAttention
from mindformers.modules.transformer import AttentionMask, TransformerDecoder, VocabEmbedding
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.base_model import BaseModel
from mindformers.mindformer_book import MindFormerBook
from mindformers.tools.logger import logger
from .cpm_bee_config import CpmBeeConfig


try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import log as logger
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode
from mindformers.modules.layers import LayerNorm, Linear, \
    _args_type_validator_check, _valid_type_checks, _valid_value_checks, \
    _check_past_none_input_none, _check_input_dtype
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config, _PipeLineConfig, OpParallelConfig, \
    _Config, _check_config, MoEParallelConfig
from mindformers.modules.transformer.moe import default_moe_config, MoE, _check_moe_config

from mindformers.tools.logger import _LogActionOnce
from mindformers.tools.utils import is_version_ge

import math
from typing import Union, Optional
import mindspore
from mindspore import ops
from mindspore import Parameter
from mindspore.common.initializer import Initializer

__all__ = ['GPT2LMHeadModel']


class CpmBeeRotaryEmbedding(nn.Cell):
    """Cpm Bee RotaryEmbedding"""
    def __init__(
        self,
        dim,
        base=10000,
        distance_scale = 1,
        dtype = mindspore.float16
    ):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
        )
        self.distance_scale = distance_scale
        self.inv_freq = Tensor(inv_freq, dtype)
        self.dtype = dtype
        self.dim = dim

        # operators
        # (b, t, c), () -> (b, t, c)
        self.mul_pos_dis = ops.Mul()
        # (b, t) -> (b, t, 1)
        self.expand_pos = ops.ExpandDims()
        # (t/2,) -> (1, t/2)
        self.expand_inv_freq = ops.ExpandDims()
        # (b, t), (c/2) -> (b, t, c/2)
        self.mul_pos_inv_freq = ops.Mul()
        # (b, t, c/2), (b, t, c/2) -> (b, t, c)
        self.cat = ops.Concat(-1)
        self.cos = ops.Cos()
        self.sin = ops.Sin()
        self.neg = ops.Neg()
        # (b, t, c) -> (b, t, c/2)
        self.stride_slice = ops.StridedSlice()
        # (b, t, c), (b, t, c) -> (b, t, c)
        self.mul = ops.Mul()
        # (b, t, c), (b, t, c) -> (b, t, c)
        self.add = ops.Add()

    def construct(self, x: Tensor, x_pos: Tensor):
        """
        Args:
            x (:obj:`Tensor` of shape ``(..., dim)``): Inputs.
            x_pos (:obj:`Tensor` of shape ``(...)``): Positions of inputs.
        """
        # (b, t, c), () -> (b, t, c)
        x_pos = self.mul_pos_dis(x_pos, self.distance_scale)
        # (b, t), (c/2) -> (b, t, c/2)
        freqs = self.mul(
            self.expand_pos(x_pos, 2).to(self.dtype),
            self.expand_inv_freq(self.inv_freq, 0)
        )
        # the same implementation as sat
        # (b, t, c/2), (b, t, c/2) -> (b, t, c)
        emb = self.cat((freqs, freqs))
        emb_cos = self.cos(emb)  # (..., dim)
        emb_sin = self.sin(emb)  # (..., dim)

        # (b, t, c/2), (b, t, c/2) -> (b, t, c)
        rotate_x = self.cat(
            [
                self.stride_slice(self.neg(x), (0, 0, 0), x.shape[:-1] + (self.dim //2,), (1, 1, 1)),
                self.stride_slice(self.neg(x), (0, 0, self.dim // 2), x.shape[:-1] + (self.dim,), (1, 1, 1))
            ]
        )  # (..., dim)

        return self.add((self.mul(x, emb_cos)), self.mul(rotate_x, emb_sin))

    def shard(self, dp, mp):
        # (b, t, c), () -> (b, t, c)
        self.mul_pos_dis.shard(((dp, 1, mp), ()))
        # (b, t) -> (b, t, 1)
        self.expand_pos.shard((dp, 1))
        # (t/2,) -> (1, t/2)
        self.expand_inv_freq.shard((1,))
        # (b, t), (c/2) -> (b, t, c/2)
        self.mul_pos_inv_freq.shard((dp, 1), (mp,))
        # (b, t, c/2), (b, t, c/2) -> (b, t, c)
        self.cat.shard((((dp, 1, mp), (dp, 1, mp))))
        self.cos.shard((dp, 1, mp))
        self.sin.shard((dp, 1, mp))
        self.neg.shard((dp, 1, mp))
        # (b, t, c) -> (b, t, c/2)
        self.stride_slice.shard((dp, 1, 1))
        # (b, t, c), (b, t, c) -> (b, t, c)
        self.mul.shard((dp, 1, mp))
        # (b, t, c), (b, t, c) -> (b, t, c)
        self.add.shard((dp, 1, mp))


class CpmBeeEmbeddingExt(nn.Cell):
    """Cpm Bee EmbeddingExt"""
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        distance_scale: int = 16,
        dtype: mindspore.tensor_type = mindspore.float16,
        param_init: Union[str, Initializer] = 'normal'
    ):

        super().__init__()

        self.rotary_emb = CpmBeeRotaryEmbedding(
            dim=embedding_size, distance_scale=distance_scale, dtype=dtype
        )

        self.weight = Parameter(initializer(param_init, (vocab_size, embedding_size), dtype),
                                'weight', parallel_optimizer=False)

        self.vocab_size = vocab_size
        self.embedding_size = Tensor(embedding_size)

        # forward
        self.gather = ops.Gather()
        self.sqrt = ops.Sqrt()
        self.div = ops.RealDiv()
        # projection
        self.matmul = ops.BatchMatMul()
        self.cat = ops.Concat(-1)

    def construct(self, ids: Tensor, ids_sub: Tensor):
        """
        Args:
            ids (:obj:`Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
            ids (:obj:`Tensor` of shape ``(batch_size)``): Subscript of input sequence tokens.
        Return:
            :obj:`Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501

        embeds = self.gather(self.weight, ids, 0)
        embeds = self.div(embeds, self.sqrt(self.embedding_size))
        return self.rotary_emb(embeds, ids_sub)

    def shard(self, dp, mp, only_dp=False):
        if only_dp:
            self.gather.shard(((1, 1), (dp, 1)))
            logger.info(f"Using {dp} data parallel for the embedding lookup.")
        else:
            if self.vocab_size % mp != 0:
                raise ValueError(f"The vocab size of the embedding {self.vocab_size} must be a "
                                 f"multiple of parallel_config.model_parallel {mp}.")

            self.gather.shard(((mp, 1), (dp, 1)))
            logger.info(f"Using {dp} data parallel and {mp} "
                        f"model parallel for the embedding lookup.")
        self.div.shard(((dp, 1, mp), ()))
        self.rotary_emb.shard(dp, mp)

        # projection
        self.matmul.shard(((dp, 1, mp), (mp, 1)))
        self.cat.shard(((dp, 1, 1), (dp, 1, 1)))

    def projection(self, x: Tensor, ext_table: Optional[Tensor] = None):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size,
        than projection map embed_size back to vocab_size.

        Args:
            x (:obj:`Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
            ext_table (:obj:`Tensor` of shape ``(ext_table_size, dim_model)``): Ext vocab table.
        Returns:
            :obj:`Tensor` of shape ``(batch, seq_len, vocab_size + ext_table_size)``: The projection output.
        """  # noqa: E501
        logits = self.matmul(self.div(x, self.sqrt(self.embedding_size)), self.weight)
        if ext_table is not None:
            logits_ext = self.matmul(x, ext_table)
            logits = self.cat([logits, logits_ext])
        return logits


class CpmBeeBucketPositionBias(nn.Cell):
    """Cpm Bee BucketPositionBias"""
    def __init__(self,
                num_heads: int,
                num_buckets: int = 32,
                num_segment_bucket: int = 32,
                max_distance: int = 128,
                dtype: mindspore.tensor_type = mindspore.float16,
                param_init: Union[str, Initializer] = 'normal'
                 ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_segment_bucket = num_segment_bucket

        self.relative_attention_bias = Parameter(initializer(param_init, (self.num_buckets + self.num_segment_bucket, num_heads), dtype),
                                                 'weight')

    def construct(
        self,
        query_pos: Tensor,  # (batch, len_q)
        key_pos: Tensor,  # (batch, len_k)
        rel_buckets: Tensor,  # (batch, len_q, len_k)
    ):
        batch = key_pos.shape[0]
        keylen = key_pos.shape[1]
        querylen = query_pos.shape[1]

        if key_pos.shape[0] != query_pos.shape[0]:
            raise AssertionError(
                f"key_pos.shape[0] should be equal to query_pos.shape[0], but got {key_pos.shape[0]} and {query_pos.shape[0]}!"
            )
        assert (
            rel_buckets.shape[0] == batch
            and rel_buckets.shape[1] == querylen
            and rel_buckets.shape[2] == keylen
        )

        relative_position_bucket = rel_buckets - 1 + self.num_buckets  # 与相对位置编码区间不重叠

        # b*q*k
        inner_segment_bucket = self._position_bucket(
            key_pos[..., None, :] - query_pos[..., :, None],
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        relative_position_bucket = ops.where(
            rel_buckets == 0,
            inner_segment_bucket,
            relative_position_bucket,
        )
        # (batch, len_q, len_k)
        relative_position_bucket = ops.stop_gradient(relative_position_bucket)

        # (batch, len_q, len_k, num_heads)
        relative_position_bucket_shape = relative_position_bucket.shape
        embeds = ops.gather(
            self.relative_attention_bias, relative_position_bucket.reshape(-1), 0
        )
        embeds = embeds.reshape(relative_position_bucket_shape + (self.num_heads,))
                # (batch, num_heads, len_q, len_k)
        embeds = embeds.transpose(0, 3, 1, 2)
        return embeds

    def _position_bucket(self, relative_position, num_buckets=32, max_distance=128):
        relative_buckets = 0
        # always bidirectional in CPMAnt
        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(mindspore.int32) * num_buckets
        relative_position = ops.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            ops.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(mindspore.int32)
        relative_postion_if_large = ops.minimum(
            relative_postion_if_large,
            ops.full_like(relative_postion_if_large, num_buckets - 1),
        )
        relative_buckets += ops.where(
            is_small, relative_position.to(mindspore.int32), relative_postion_if_large
        )
        return relative_buckets


class CpmDenseGatedACT(nn.Cell):
    """Cpm Ant dense gated act"""
    def __init__(self, hidden_size, dim_ff, param_init_type=mstype.float32):
        super().__init__()

        self.w_0 = Linear(in_channels=hidden_size,
                          out_channels=dim_ff,
                          has_bias=False,
                          transpose_b=False,
                          param_init_type=param_init_type)
        self.w_1 = Linear(in_channels=hidden_size,
                          out_channels=dim_ff,
                          has_bias=False,
                          transpose_b=False,
                          param_init_type=param_init_type)

        self.mul = P.Mul()
        self.act = nn.GELU(False)

    def construct(self, hidden_states: Tensor):
        """Transform an input tensor from one feature space to another via a nonlinear operation

        Args:
            hidden_states (`Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        gate_score = self.act(self.w_0(hidden_states))
        hidden_states = self.w_1(hidden_states)

        hidden_states = self.mul(gate_score, hidden_states)
        return hidden_states

    def shard(self, dp, mp):
        self.w_0.shard(strategy_matmul=((dp, 1), (1, mp)))
        self.w_1.shard(strategy_matmul=((dp, 1), (1, mp)))
        self.mul.shard(((dp, mp), (dp, mp)))
        self.act.shard(((dp, mp),))



class CpmAntFeedForward(nn.Cell):
    """Cpm Ant feedforward"""
    def __init__(self, config: CpmBeeConfig):
        super().__init__()

        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel

        self.w_in = CpmDenseGatedACT(config.embedding_size,
                                        config.embedding_size * config.expand_ratio,
                                        config.param_init_type,
                                        config.parallel_config)
        if config.dropout_prob is not None:
            self.dropout = Dropout(1-config.dropout_prob)
        else:
            self.dropout = None

        self.w_out = Linear(in_channels=config.embedding_size * config.expand_ratio,
                            out_channels=config.embedding_size,
                            transpose_b=False,
                            has_bias=False,
                            outer_batch=config.parallel_config.data_parallel,
                            param_init_type=config.param_init_type)

        self.w_out.shard(strategy_matmul=((dp, mp), (mp, 1)))
        self.dropout.dropout.shard(((dp, 1),))

    def construct(self, hidden_states: Tensor):
        """
        Args:
            hidden_states (`Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        hidden_states = self.w_in(hidden_states)

        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.w_out(hidden_states)

        return hidden_states


class CpmAntFFNBlock(nn.Cell):
    """Cpm Ant ffn block"""
    def __init__(self, config: CpmBeeConfig):
        super().__init__()
        dp = config.parallel_config.data_parallel
        self.layernorm_before_ffn = CpmLayerNorm([config.embedding_size], param_init_type=config.param_init_type)
        self.ffn = CpmAntFeedForward(config)
        if config.dropout_prob:
            self.dropout = nn.Dropout(1-config.dropout_prob)
        else:
            self.dropout = None
        self.dropout.dropout.shard(((dp, 1),))

    def construct(
        self,
        hidden_states: Tensor,
    ):
        """
        Args:
            hidden_states (`Tensor` of shape `(batch, len_seq, dim_model)`):
                Hidden states before feed forward layer.
        """
        ln_outputs = self.layernorm_before_ffn(hidden_states)
        outputs = self.ffn(ln_outputs)
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        hidden_states = hidden_states + outputs
        return hidden_states

# class CpmTransformerBlock(nn.Cell):
#     @_LogActionOnce(m_logger=logger, key='CpmTransformerBlock',
#                     no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
#     @_args_type_validator_check(hidden_size=Validator.check_positive_int,
#                                 num_heads=Validator.check_positive_int,
#                                 ffn_hidden_size=Validator.check_positive_int,
#                                 seq_length=Validator.check_positive_int,
#                                 attention_dropout_rate=Validator.check_non_negative_float,
#                                 hidden_dropout_rate=Validator.check_non_negative_float,
#                                 post_layernorm_residual=Validator.check_bool,
#                                 layernorm_compute_type=_valid_value_checks([mstype.float32, mstype.float16],
#                                                                            "CpmTransformerBlock"),
#                                 softmax_compute_type=_valid_value_checks([mstype.float32, mstype.float16],
#                                                                          "CpmTransformerBlock"),
#                                 param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
#                                                                     "CpmTransformerBlock"),
#                                 parallel_config=_valid_type_checks([OpParallelConfig, MoEParallelConfig],
#                                                                    "CpmTransformerBlock"),
#                                 use_past=Validator.check_bool)
#     def __init__(self,
#                  config,
#                  batch_size,
#                  hidden_size,
#                  ffn_hidden_size,
#                  num_heads,
#                  seq_length,
#                  attention_dropout_rate=0.1,
#                  hidden_dropout_rate=0.1,
#                  post_layernorm_residual=False,
#                  layernorm_compute_type=mstype.float32,
#                  softmax_compute_type=mstype.float32,
#                  param_init_type=mstype.float32,
#                  hidden_act='gelu',
#                  use_past=False,
#                  moe_config=default_moe_config,
#                  parallel_config=default_dpmp_config):
#         super().__init__()
#         if batch_size or use_past:
#             Validator.check_positive_int(batch_size)
#         self.batch_size = batch_size
#         if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
#             _check_config(parallel_config)
#             if num_heads % parallel_config.model_parallel != 0:
#                 raise ValueError(
#                     "For 'TransformerEncoderLayer', the class variable 'num_heads' must be divisibled by the "
#                     "'parallel_config.model_parallel', but got the num_heads is {} and "
#                     "parallel_config.model_parallel is {}.".format(num_heads, parallel_config.model_parallel))
#             if hidden_size % parallel_config.model_parallel != 0:
#                 raise ValueError(
#                     "For 'TransformerEncoderLayer', the class variable 'hidden_size' must be divisibled by "
#                     "the 'parallel_config.model_parallel', but got the hidden_size is {} and parallel_config."
#                     " model_parallel is {}.".format(hidden_size, parallel_config.model_parallel))
#             if ffn_hidden_size % parallel_config.model_parallel != 0:
#                 raise ValueError(
#                     "For 'TransformerEncoderLayer', the class variable 'ffn_hidden_size' must be divisibled "
#                     "by the 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
#                     "and parallel_config. model_parallel is {}."
#                     .format(ffn_hidden_size, parallel_config.model_parallel))
#             _check_moe_config(moe_config, parallel_config)
#             self.use_moe = (moe_config.expert_num > 1)
#             self.use_past = use_past
#             self.seq_length = seq_length
#             self.hidden_size = hidden_size
#             self.layernorm1 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
#             self.layernorm2 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)

#             attention_parallel_config = parallel_config.dpmp if self.use_moe else parallel_config
#             self.attention = CpmAttention(batch_size=batch_size,
#                                         src_seq_length=seq_length,
#                                         tgt_seq_length=seq_length,
#                                         hidden_size=hidden_size,
#                                         num_heads=num_heads,
#                                         hidden_dropout_rate=hidden_dropout_rate,
#                                         attention_dropout_rate=attention_dropout_rate,
#                                         softmax_compute_type=softmax_compute_type,
#                                         param_init_type=param_init_type,
#                                         use_past=use_past,
#                                         parallel_config=attention_parallel_config)

#             self.output = CpmAntFFNBlock(config)
#             self.post_layernorm_residual = post_layernorm_residual
#             self.add = P.Add().shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1)))
#             self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
#             self.dtype = mstype.float16
#             self.key_past = None
#             self.value_past = None

#             if self.use_past:
#                 # operator used for state reuse
#                 self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
#                 self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
#                 self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
#                 size_per_head = hidden_size // num_heads
#                 self.key_shape = (batch_size, num_heads, size_per_head, seq_length)
#                 self.value_shape = (batch_size, num_heads, seq_length, size_per_head)
#                 # parameters saving key and value states
#                 self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
#                 self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
#                 self.tile = P.Tile().shard(((1, 1),))
#                 self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
#                 self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
#         elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
#             _check_config(parallel_config)
#             if num_heads % parallel_config.model_parallel != 0:
#                 raise ValueError(
#                     "For 'TransformerEncoderLayer', the class variable 'num_heads' must be divisibled by the "
#                     "'parallel_config.model_parallel', but got the num_heads is {} and "
#                     "parallel_config.model_parallel is {}.".format(num_heads, parallel_config.model_parallel))
#             if hidden_size % parallel_config.model_parallel != 0:
#                 raise ValueError(
#                     "For 'TransformerEncoderLayer', the class variable 'hidden_size' must be divisibled by "
#                     "the 'parallel_config.model_parallel', but got the hidden_size is {} and parallel_config."
#                     " model_parallel is {}.".format(hidden_size, parallel_config.model_parallel))
#             if ffn_hidden_size % parallel_config.model_parallel != 0:
#                 raise ValueError(
#                     "For 'TransformerEncoderLayer', the class variable 'ffn_hidden_size' must be divisibled "
#                     "by the 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
#                     "and parallel_config. model_parallel is {}."
#                     .format(ffn_hidden_size, parallel_config.model_parallel))
#             _check_moe_config(moe_config, parallel_config)
#             self.use_moe = (moe_config.expert_num > 1)
#             self.use_past = use_past
#             self.seq_length = seq_length
#             self.hidden_size = hidden_size
#             self.layernorm1 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
#             self.layernorm1.shard(((parallel_config.data_parallel, 1),))
#             self.layernorm2 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
#             self.layernorm2.shard(((parallel_config.data_parallel, 1),))

#             attention_parallel_config = parallel_config.dpmp if self.use_moe else parallel_config
#             self.attention = MultiHeadAttention(batch_size=batch_size,
#                                                 src_seq_length=seq_length,
#                                                 tgt_seq_length=seq_length,
#                                                 hidden_size=hidden_size,
#                                                 num_heads=num_heads,
#                                                 hidden_dropout_rate=hidden_dropout_rate,
#                                                 attention_dropout_rate=attention_dropout_rate,
#                                                 softmax_compute_type=softmax_compute_type,
#                                                 param_init_type=param_init_type,
#                                                 use_past=use_past,
#                                                 parallel_config=attention_parallel_config)

#             # Feed Forward Network, FFN
#             self.output = FeedForward(hidden_size=hidden_size,
#                                         dropout_rate=hidden_dropout_rate,
#                                         ffn_hidden_size=ffn_hidden_size,
#                                         param_init_type=param_init_type,
#                                         hidden_act=hidden_act,
#                                         parallel_config=parallel_config)
#             self.post_layernorm_residual = post_layernorm_residual
#             self.add = P.Add().shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1)))
#             self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
#             self.dtype = mstype.float16
#             self.key_past = None
#             self.value_past = None

#             if self.use_past:
#                 # operator used for state reuse
#                 self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
#                 self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
#                 self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
#                 size_per_head = hidden_size // num_heads
#                 self.key_shape = (batch_size, num_heads, size_per_head, seq_length)
#                 self.value_shape = (batch_size, num_heads, seq_length, size_per_head)
#                 # parameters saving key and value states
#                 self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
#                 self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
#                 self.tile = P.Tile().shard(((1, 1),))
#                 self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
#                 self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
#         else:
#             raise RuntimeError(f"The {self.cls_name} only support sharding propagation or "
#                                f"semi-auto parallel mode now.")

#     def construct(self, x, input_mask=None, init_reset=True, batch_valid_length=None):
#         """forward process"""
#         self._check_input(x, input_mask, init_reset, batch_valid_length)
#         x_shape = F.shape(x)
#         x = F.reshape(x, (-1, x_shape[-1]))
#         if self.post_layernorm_residual:
#             input_x = x
#         else:
#             input_x = self.layernorm1(x)
#         input_x = F.cast(input_x, self.dtype)

#         # indicate whether reset saved states
#         key_reset = None
#         value_reset = None

#         if self.use_past:
#             # reset states, init_reset True for reuse and False for reset
#             self.assign(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
#             key_reset = self.key_past
#             self.assign(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
#             value_reset = self.value_past
#             # add dependency for desired execution order
#             input_x = F.depend(input_x, key_reset)
#             input_x = F.depend(input_x, value_reset)

#         attention, layer_present = self.attention(input_x, input_x, input_x, input_mask,
#                                                   self.key_past, self.value_past, batch_valid_length)
#         # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
#         if self.post_layernorm_residual:
#             x = self.add(input_x, attention)
#         # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
#         else:
#             x = self.add(x, attention)

#         output_x = self.layernorm2(x)
#         output_x = F.cast(output_x, self.dtype)
#         aux_loss = None
#         if self.use_moe:
#             mlp_logit, aux_loss = self.output(output_x)
#         else:
#             mlp_logit = self.output(output_x)

#         value_update = None
#         key_update = None
#         if self.use_past:
#             # current key and value
#             key_present, value_present = layer_present
#             # update key and value calculated this step
#             self.assign(self.key_past, key_present)
#             key_update = self.key_past
#             self.assign(self.value_past, value_present)
#             value_update = self.value_past
#             # add dependency for desired execution order
#             key_update = F.depend(key_update, key_reset)
#             value_update = F.depend(value_update, value_reset)

#         # add dependency for desired execution order
#         mlp_logit = F.depend(mlp_logit, value_update)
#         mlp_logit = F.depend(mlp_logit, key_update)

#         # if shape is 3d, we reshape the inputs of the add
#         if len(x_shape) == 3:
#             output_x = P.Reshape()(output_x, x_shape)
#             mlp_logit = P.Reshape()(mlp_logit, x_shape)
#             x = P.Reshape()(x, x_shape)

#             if self.post_layernorm_residual:
#                 output = self.add_3d(output_x, mlp_logit)
#                 output = F.reshape(output, (-1, x_shape[-1]))
#                 output = self.layernorm1(output)
#                 output = F.reshape(output, x_shape)
#             else:
#                 output = self.add_3d(x, mlp_logit)
#         else:
#             if self.post_layernorm_residual:
#                 output = self.add(output_x, mlp_logit)
#                 output = self.layernorm1(output)
#             else:
#                 output = self.add(x, mlp_logit)
#             output = F.reshape(output, x_shape)

#         if self.use_moe:
#             return output, layer_present, aux_loss
#         return output, layer_present

#     def _check_input(self, x, input_mask, init_reset, batch_valid_length):
#         r"""Check inputs"""
#         _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16], self.cls_name)
#         if input_mask is not None:
#             _check_input_dtype(F.dtype(input_mask), "input_mask", [mstype.float32, mstype.float16], self.cls_name)

#         init_reset_is_tensor = isinstance(init_reset, Tensor)
#         init_reset_is_default = init_reset is True
#         batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
#         batch_is_default = batch_valid_length is None
#         _check_past_none_input_none(self.use_past, "init_reset", self.cls_name, True, init_reset_is_tensor,
#                                     init_reset_is_default)
#         _check_past_none_input_none(self.use_past, "batch_valid_length", self.cls_name, None,
#                                     batch_valid_length_is_tensor, batch_is_default)

#         if self.use_past:
#             _check_input_dtype(F.dtype(init_reset), "init_reset", [mstype.bool_], self.cls_name)
#             _check_input_dtype(F.dtype(batch_valid_length), "batch_valid_length", [mstype.int32], self.cls_name)
#         return True


# @MindFormerRegister.register(MindFormerModuleType.MODELS)
# class GPT2LMHeadModel(BaseModel):
#     r"""
#         Provide gpt training loss or logits through network.
#         Args:
#             config (GPT2Config): The config of Gpt2Model.

#         Returns:
#             Tensor, the loss or logits of the network.
#         """
#     _support_list = MindFormerBook.get_model_support_list()['gpt2']

#     def __init__(self, config: GPT2Config = None):
#         config = config if config is not None else GPT2Config()
#         super(GPT2LMHeadModel, self).__init__(config, auto_prefix=False)

#         self.eos_token = self.config.eos_token
#         parallel_config = self.config.parallel_config
#         self.stridedslice = P.StridedSlice().shard(((parallel_config.data_parallel, 1),))
#         self.not_equal = P.NotEqual().shard(((parallel_config.data_parallel, 1), ()))

#         self.get_attention_mask = AttentionMask(seq_length=config.seq_length,
#                                                 parallel_config=parallel_config.dp_mp_config)

#         self.backbone = GPT2Model(config)
#         self.head = GPTHead(hidden_size=config.embedding_size,
#                             vocab_size=config.vocab_size,
#                             parallel_config=self.config.parallel_config)
#         if parallel_config.pipeline_stage > 1:
#             self.head.pipeline_stage = parallel_config.pipeline_stage - 1
#             self.backbone.embedding.word_embedding.embedding_table.add_pipeline_stage(self.head.pipeline_stage)

#         mp = config.parallel_config.model_parallel
#         vocab_size = config.vocab_size
#         loss_parallel_config = copy.deepcopy(parallel_config)
#         if vocab_size % mp != 0:
#             logger.warning("The vocab size of GPT Loss is: %s, it is not divide by model_parallel: %s",
#                            vocab_size, mp)
#             logger.warning("Now, the model_parallel num of GPT Loss will be changed: mp = 1")
#             loss_parallel_config.model_parallel = 1

#         self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config)
#         self.reshape = P.Reshape()
#         self.cast = P.Cast()
#         self.load_checkpoint(config)
#         self.add = P.Add().shard(((parallel_config.data_parallel, 1), ()))

#     def construct(self, input_ids):
#         r"""
#             construct function for Language Modeling

#             Args:
#                 input_ids (Tensor): the indices of input sequence tokens in the vocabulary.

#             Returns:
#                 logits (Tensor) or loss (mstype.float32): if is_training is False, directly return the logits,
#                                                          otherwise, return the computed loss.
#         """

#         batch_size, seq_length = input_ids.shape

#         if self.phase == "train":
#             tokens = self.stridedslice(input_ids, (0, 0), (batch_size, seq_length - 1), (1, 1))
#         else:
#             tokens = input_ids

#         input_mask = self.not_equal(tokens, self.eos_token)
#         input_mask = self.cast(input_mask, mstype.float32)
#         attention_mask = self.get_attention_mask(input_mask)

#         # [batch_size, seq_length, vocab_size]
#         output_states, embedding_table = self.backbone(tokens, attention_mask)
#         logits = self.head(output_states, embedding_table)

#         if self.phase != 'train':
#             logits = self.reshape(logits, (batch_size, seq_length, -1))

#             # makes cast effective to avoid allgather issue in Mindspore1.10
#             input_mask = self.add(input_mask, 1)

#             return logits, tokens, input_mask

#         labels = self.stridedslice(input_ids, (0, 1), (batch_size, seq_length), (1, 1))
#         labels = self.reshape(labels, (-1,))
#         input_mask = self.reshape(input_mask, (-1,))

#         loss = self.loss(logits, labels, input_mask)
#         return loss


# class GPTEmbeddingLayer(nn.Cell):
#     r"""The Embedding Layer of GPT-2 network."""

#     def __init__(self, config: GPT2Config = None):
#         super(GPTEmbeddingLayer, self).__init__()
#         parallel_config = copy.deepcopy(config.parallel_config)
#         embedding_mp = config.parallel_config.embedding_dp_mp_config.model_parallel
#         vocab_size = config.vocab_size
#         if vocab_size % embedding_mp != 0:
#             logger.warning("The vocab size of embedding layer is: %s, it is not divide by model_parallel: %s",
#                            vocab_size, embedding_mp)
#             logger.warning("Now, model_parallel will be changed: mp = 1")
#             parallel_config.embedding_dp_mp_config.model_parallel = 1

#         self.word_embedding = VocabEmbedding(vocab_size=vocab_size,
#                                              embedding_size=config.embedding_size,
#                                              param_init=initializer(TruncatedNormal(config.initializer_range),
#                                                                     [vocab_size, config.embedding_size],
#                                                                     dtype=mstype.float32),
#                                              parallel_config=parallel_config.embedding_dp_mp_config)
#         new_parallel_config = copy.deepcopy(parallel_config)
#         new_parallel_config.vocab_emb_dp = True

#         self.position_embedding = VocabEmbedding(vocab_size=config.seq_length,
#                                                  embedding_size=config.embedding_size,
#                                                  param_init=initializer(TruncatedNormal(config.initializer_range),
#                                                                         [config.seq_length, config.embedding_size],
#                                                                         dtype=mstype.float32),
#                                                  parallel_config=new_parallel_config.embedding_dp_mp_config)
#         self.add = P.Add().shard(
#             ((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
#         self.dropout = Dropout(1 - config.dropout_prob)
#         self.dropout.shard(((parallel_config.data_parallel, 1, 1),))

#     def construct(self, input_ids, input_position):
#         """The forward compute of Embedding Layer."""
#         word_embedding, word_table = self.word_embedding(input_ids)
#         position_embedding, _ = self.position_embedding(input_position)
#         embedding = self.add(word_embedding, position_embedding)
#         embedding = self.dropout(embedding)
#         return embedding, word_table


# def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
#     r"""
#         Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

#         Args:
#             network(Cell) - Represents the transformer block
#             parallel_config(dict) - Parallel Config
#             layer_id(int) - Means the layer index for the current module, counts from zero.
#             offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
#             layers(int) - The total layers used for the model.
#     """
#     pp_dis = max(int((layers + 1) / parallel_config.pipeline_stage), 1)
#     pp_id = min((layer_id + offset) // pp_dis, parallel_config.pipeline_stage - 1)
#     network.pipeline_stage = pp_id

#     # Used for optimizer's fusion tag
#     dis = max(int((layers + 1) / parallel_config.gradient_aggregation_group), 1)
#     if parallel_config.pipeline_stage > 1:
#         network.set_comm_fusion(2)
#     else:
#         network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
#     if isinstance(parallel_config.recompute, bool):
#         if parallel_config.recompute:
#             network.recompute()
#     else:
#         if parallel_config.recompute.recompute:
#             network.recompute(recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)


# class GPT2Model(nn.Cell):
#     """
#     The backbone of GPT network

#     Args:
#         config(GPT2Config): the config of network

#     Inputs:
#         input_ids: the tokenized inputs with datatype int32
#         input_mask: the mask indicating whether each position is a valid input

#     Returns:
#         output_state: Tensor, the output logit of backbone
#         present_layer: Tensor, the current feature map
#         embedding_table: Tensor, the embedding table for the vocabulary
#     """

#     def __init__(self, config):
#         super(GPT2Model, self).__init__()

#         self.embedding = GPTEmbeddingLayer(config)
#         self.embedding.pipeline_stage = 0

#         self.layernorm = LayerNorm((config.embedding_size,)).to_float(config.layernorm_dtype)
#         if config.parallel_config.pipeline_stage > 1:
#             self.layernorm.set_comm_fusion(2)
#         else:
#             self.layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
#         self.layernorm.shard(((config.parallel_config.data_parallel, 1),))
#         self.layernorm.pipeline_stage = config.parallel_config.pipeline_stage - 1

#         if not hasattr(config.parallel_config, "moe_config"):
#             config.parallel_config.moe_config = default_moe_config
#         moe_config = config.parallel_config.moe_config
#         self.blocks = TransformerDecoder(hidden_size=config.embedding_size,
#                                          batch_size=config.batch_size,
#                                          ffn_hidden_size=config.embedding_size * config.expand_ratio,
#                                          src_seq_length=config.seq_length,
#                                          tgt_seq_length=config.seq_length,
#                                          num_layers=config.num_layers,
#                                          num_heads=config.num_heads,
#                                          attention_dropout_rate=config.attention_probs_dropout_prob,
#                                          hidden_dropout_rate=config.hidden_dropout_prob,
#                                          hidden_act=config.hidden_act,
#                                          lambda_func=set_parallel_configure_for_layer,
#                                          param_init_type=config.param_init_type,
#                                          layernorm_compute_type=config.layernorm_dtype,
#                                          softmax_compute_type=config.softmax_dtype,
#                                          parallel_config=config.parallel_config,
#                                          moe_config=moe_config).blocks
#         self.cast = P.Cast()
#         self.tile = P.Tile().shard(((config.parallel_config.data_parallel,),))
#         self.dtype = mstype.float16
#         self.num_layers = config.num_layers
#         self.input_position = Tensor(np.arange(config.seq_length), mstype.int32)

#     def construct(self, input_ids, attention_mask):
#         """GPT model"""
#         batch_size, _ = F.shape(input_ids)
#         input_position = self.tile(self.input_position, (batch_size, 1))

#         input_embedding, embedding_table = self.embedding(input_ids, input_position)

#         hidden_states = self.cast(input_embedding, self.dtype)
#         hidden_shape = F.shape(hidden_states)
#         hidden_states = F.reshape(hidden_states, (-1, hidden_shape[-1]))

#         for i in range(self.num_layers):
#             hidden_states, _ = self.blocks[i](hidden_states, attention_mask)

#         output_state = self.layernorm(hidden_states)

#         return output_state, embedding_table


# class GPTHead(nn.Cell):
#     r"""Head for GPT to get the logits of each token in the vocab."""

#     def __init__(self,
#                  hidden_size,
#                  vocab_size,
#                  compute_type=mstype.float16,
#                  parallel_config=None):
#         super().__init__()
#         copied_parallel_config = copy.deepcopy(parallel_config)
#         mp = copied_parallel_config.model_parallel
#         if vocab_size % mp != 0:
#             logger.warning("The vocab size of GPTHead MatMul is: %s, it is not divide by model_parallel: %s",
#                            vocab_size, mp)
#             logger.warning("Now, the model_parallel num of GPTHead MatMul will be changed: mp = 1")
#             copied_parallel_config.model_parallel = 1

#         if copied_parallel_config.pipeline_stage > 1:
#             copied_parallel_config.vocab_emb_dp = False
#         if copied_parallel_config.vocab_emb_dp:
#             self.matmul = P.MatMul(transpose_b=True).shard(((copied_parallel_config.data_parallel, 1), (1, 1)))
#         else:
#             self.matmul = P.MatMul(transpose_b=True).shard(((copied_parallel_config.data_parallel, 1), (
#                 copied_parallel_config.model_parallel, 1)))
#         self.hidden_size = hidden_size
#         self.dtype = compute_type
#         self.cast = P.Cast()

#     def construct(self, state, embedding_table):
#         logits = self.matmul(self.cast(state, self.dtype), self.cast(embedding_table, self.dtype))
#         return logits


# class CrossEntropyCalculationWithMask(nn.Cell):
#     """
#     Cross Entropy loss
#     """

#     def __init__(self, is_training=None, num_labels=None):
#         super(CrossEntropyCalculationWithMask, self).__init__()
#         self.onehot = P.OneHot()
#         self.on_value = Tensor(1.0, mstype.float32)
#         self.off_value = Tensor(0.0, mstype.float32)
#         self.reduce_sum = P.ReduceSum()
#         self.reduce_mean = P.ReduceMean()
#         self.reshape = P.Reshape()
#         self.last_idx = (-1,)
#         self.neg = P.Neg()
#         self.cast = P.Cast()
#         self.is_training = is_training
#         self.num_labels = num_labels
#         self.log_softmax = P.LogSoftmax(axis=-1)

#     def construct(self, logits, label_ids, input_mask=None):
#         """
#         Calculate loss

#         Args:
#             logits (Tensor): the probability distribution over vocabulary.
#             label_ids (Tensor): the indices of input sequence tokens in the vocabulary.
#             input_mask (Tensor): input sentences padding mask, where 0 indicates padding position.

#         Returns:
#             return_value (Tensor, mstype.float32): if is_training is False, directly return the logits, otherwise,
#                                                    return the computed loss.
#         """

#         # logits [batch * (seq_length-1), vocab_size]   label_ids [batch, seq_length-1]
#         logits = self.log_softmax(logits)

#         if self.is_training:
#             label_ids = self.reshape(label_ids, self.last_idx)  # label_ids [batch * (seq_length-1)]
#             one_hot_labels = self.onehot(label_ids, self.num_labels, self.on_value,
#                                          self.off_value)  # [batch * (seq_length-1), vocab_size]
#             per_example_loss = self.neg(
#                 self.reduce_sum(one_hot_labels * logits, self.last_idx))  # [batch * (seq_length-1)]

#             # for PPL calculation in evaluation
#             if input_mask is not None:
#                 input_mask = self.cast(self.reshape(input_mask, self.last_idx),
#                                        mstype.float32)  # [batch * (seq_length-1)]

#                 valid_loss_sum = self.reduce_sum(input_mask * per_example_loss, ())
#                 valid_element_sum = self.reduce_sum(input_mask, ()) + self.cast(F.tuple_to_array((1e-5,)),
#                                                                                 mstype.float32)
#                 loss = valid_loss_sum / valid_element_sum
#             else:
#                 loss = self.reduce_mean(per_example_loss, self.last_idx)  # a number
#             return_value = self.cast(loss, mstype.float32)
#         else:
#             return_value = logits * 1.0  # [batch * (seq_length-1), vocab_size]

#         return return_value
