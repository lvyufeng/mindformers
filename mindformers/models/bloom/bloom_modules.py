"""
Note:
    Transformer Networks. This is interface that is subject to change or deletion.
    直接复制需要的类改，比较简单粗暴。FFN 只改了初始化逻辑，可以继承。
"""
from __future__ import absolute_import

import math

import mindspore.common.dtype as mstype
import numpy as np
from mindformers.modules.layers import Linear, LayerNorm, _check_input_shape, \
    _check_shape_equal, _check_past_none_input_none, _check_input_dtype
from mindspore import context
from mindspore import nn
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import Uniform, Normal
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.nn.cell import Cell
from mindspore.nn.transformer.moe import default_moe_config, _check_moe_config
from mindspore.nn.transformer.op_parallel_config import default_dpmp_config, _check_config
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode
from .initializer import F2StdNormal

class AlibiTensor(nn.Cell):
    def __init__(self, seq_length, num_heads):
        super().__init__()
        self.seq_length = seq_length
        self.num_heads = num_heads

        # build slopes
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base = np.array(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=np.float32)
        powers = np.arange(1, 1 + closest_power_of_2, dtype=np.int32)
        slopes = np.power(base, powers)

        if closest_power_of_2 != num_heads:
            extra_base = np.array(
                2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=np.float32
            )
            num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
            extra_powers = np.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=np.int32)
            slopes = np.concatenate([slopes, np.power(extra_base, extra_powers)], axis=0)

        self.slopes = Tensor(slopes, mstype.float32)

    def construct(self, attention_mask, dtype):
        """
        Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
        relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
        `softmax(l+a) = softmax(l)`. Based on
        https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
        TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

        Args:
        Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
            attention_mask:
                Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
            num_heads:
                number of heads
            dtype:
                dtype of the output tensor
        """
        batch_size = attention_mask.shape[0]

        # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
        # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
        # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
        # => the query_length dimension will then be broadcasted correctly
        # This is more or less identical to T5's relative position bias:
        # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
        arange_tensor = ((attention_mask.cumsum(axis=-1) - 1) * attention_mask)[:, None, :]
        alibi = self.slopes[..., None] * arange_tensor
        return alibi.reshape(batch_size, self.num_heads, 1, self.seq_length).astype(dtype)


class BloomFeedForward(nn.Cell):
    def __init__(self, 
                 config,
                 hidden_size,
                 ffn_hidden_size,
                 dropout_rate,
                 hidden_act='gelu',
                 expert_num=1,
                 expert_group_size=None,
                 in_param_init_func=None,
                 out_param_init_func=None,
                 param_init_type=mstype.float32,
                 parallel_config=default_dpmp_config):
        super().__init__()
        # add wang init
        in_weight_init = Normal(config.initializer_range)
        out_weight_init = F2StdNormal(config.num_layers, config.initializer_range)
        bias_init = 'zeros'

        _check_config(parallel_config)

        mp = parallel_config.model_parallel
        dp = parallel_config.data_parallel

        if ffn_hidden_size % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'ffn_hidden_size' must be a multiple of the"
                             "num of model parallel, but got the ffn_hidden_size is {} and the num of model "
                             "parallel is {}.".format(ffn_hidden_size, mp))
        if hidden_size % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'hidden_size' must be a multiple of the num of "
                             "model parallel, but got the hidden_size is {} and the num of model parallel is {}."
                             .format(hidden_size, mp))
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError("For 'FeedForward', the class variable 'dropout_rate' must be in the range [0, 1.0), "
                             "but got the value : {}.".format(dropout_rate))
        input_size = hidden_size
        output_size = ffn_hidden_size

        # Project to ffn_hidden_size
        self.mapping = Linear(in_channels=input_size,
                              out_channels=output_size,
                              weight_init=in_weight_init,  # add wang init
                              bias_init=bias_init,  # add wang init
                              activation=hidden_act,
                              transpose_b=False,
                              expert_num=expert_num,
                              expert_group_size=expert_group_size,
                              outer_batch=dp,
                              param_init_type=param_init_type)

        self.mapping.shard(strategy_matmul=((dp, 1), (1, mp)),
                           strategy_bias=((dp, mp), (mp,)),
                           strategy_activation=((dp, mp),))

        # Project back to hidden_size
        self.projection = Linear(in_channels=output_size,
                                 out_channels=input_size,
                                 weight_init=out_weight_init,  # add wang init
                                 bias_init=bias_init,  # add wang init
                                 transpose_b=False,
                                 expert_num=expert_num,
                                 expert_group_size=expert_group_size,
                                 outer_batch=dp,
                                 param_init_type=param_init_type)
        self.projection.shard(strategy_matmul=((dp, mp), (mp, 1)),
                              strategy_bias=((dp, 1), (1,)))
        self.projection.bias.parallel_optimizer = False

        self.dropout = nn.Dropout(1 - dropout_rate)
        self.dropout.dropout.shard(((dp, 1),))
        self.dropout_3d = nn.Dropout(1 - dropout_rate)
        self.dropout_3d.dropout.shard(((dp, 1, 1),))
        self.cast = P.Cast()

    def construct(self, x):
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16], self.cls_name)
        x = self.cast(x, mstype.float16)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        hidden = self.mapping(x)
        output = self.projection(hidden)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        if len(F.shape(output)) == 3:
            output = self.dropout_3d(output)
        elif len(F.shape(output)) == 2:
            output = self.dropout(output)
        return output


class BloomAttention(Cell):
    def __init__(self, config,
                 batch_size,
                 src_seq_length,
                 tgt_seq_length,
                 hidden_size,
                 num_heads,
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 parallel_config=default_dpmp_config,
                 use_relative_positions=False):
        super().__init__()
        # ROPE
        self.use_relative_positions = use_relative_positions

        self._is_ascend = context.get_context('device_target') in ["Ascend"]
        _check_config(parallel_config)
        self.is_parallel_mode = _get_parallel_mode() in (
            ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        self.src_seq_length = src_seq_length
        self.tgt_seq_length = tgt_seq_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        if hidden_dropout_rate < 0 or hidden_dropout_rate >= 1:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_dropout_rate' must be "
                             "in range [0, 1.0), but got the value : {}.".format(hidden_dropout_rate))
        if attention_dropout_rate < 0 or attention_dropout_rate >= 1:
            raise ValueError("For 'MultiHeadAttention', the class variable 'attention_dropout_rate' must be "
                             "in range [0, 1.0), but got the value : {}.".format(attention_dropout_rate))
        if hidden_size % num_heads != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'num_heads', but got the hidden_size is {} and the num_heads is {}."
                             .format(hidden_size, num_heads))
        if num_heads % parallel_config.model_parallel != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'num_heads' must be a multiple of "
                             "'parallel_config.model_parallel', but got the num_heads is {} "
                             "and the parallel_config.model_parallel  is {}."
                             .format(num_heads, parallel_config.model_parallel))
        self.is_first_iteration = True
        # Output layer
        self.projection = Linear(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 weight_init=F2StdNormal(config.num_layers, config.initializer_range),
                                 bias_init='zeros',
                                 transpose_b=False,
                                 compute_dtype=compute_dtype,
                                 param_init_type=param_init_type)
        self.projection.shard(strategy_bias=((parallel_config.data_parallel, 1), (1,)),
                              strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                               (parallel_config.model_parallel, 1)))
        self.projection.bias.parallel_optimizer = False

        self.transpose = P.Transpose().shard(
            ((parallel_config.data_parallel, 1, parallel_config.model_parallel, 1),))
        self.merger_head_transpose = P.Transpose().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1,  1),))

        self.reshape = P.Reshape()
        self.n_head = num_heads
        # embedding size per head
        self.size_per_head = hidden_size // self.n_head
        self.concat_k = P.Concat(axis=3)
        self.concat_v = P.Concat(axis=2)
        self.multiply_data = Tensor([
            -10000.0,
        ], dtype=softmax_compute_type)
        self.batch_matmul = P.BatchMatMul().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.real_div = P.RealDiv().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1), ()))
        self.sub = P.Sub().shard(
            ((1,), (parallel_config.data_parallel, 1, 1, 1)))
        self.mul = P.Mul().shard(
            ((parallel_config.data_parallel, 1, 1, 1), (1,)))
        self.add = P.Add().shard(
            ((parallel_config.data_parallel, 1, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.mul_alibi = P.Mul().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1), (1,)))
        self.add_alibi = P.Add().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))

        # Normalize factor for attention, sqrt(dk) as widely used
        self.scale_factor = Tensor(math.sqrt(math.sqrt(self.size_per_head)))
        self.inv_norm_factor = Tensor([1.0 / math.sqrt(self.size_per_head)])
        self.beta = Tensor([1.0])
        self.use_past = use_past
        self.dropout = nn.Dropout(1 - hidden_dropout_rate)
        self.dropout.dropout.shard(((parallel_config.data_parallel, 1),))
        self.prob_dropout = nn.Dropout(1 - attention_dropout_rate)
        self.prob_dropout.dropout.shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
        self.softmax = nn.Softmax().to_float(softmax_compute_type)
        self.softmax.softmax.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
        self.softmax_3d = nn.Softmax().to_float(softmax_compute_type)
        self.softmax_3d.softmax.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1),))
        self.expand_dims = P.ExpandDims().shard(((parallel_config.data_parallel, 1, 1),))

        # Query
        bound = 1 / math.sqrt(hidden_size)
        bias_init = Uniform(bound)
        in_weight_init = Normal(config.initializer_range)
        self.dense1 = Linear(hidden_size,
                             hidden_size,
                             weight_init=in_weight_init,
                             bias_init=bias_init,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
        self.dense1.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                          strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                         (parallel_config.model_parallel,)))
        # Key
        self.dense2 = Linear(hidden_size,
                             hidden_size,
                             weight_init=in_weight_init,
                             bias_init=bias_init,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
        self.dense2.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                          strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                         (parallel_config.model_parallel,)))

        # Value
        self.dense3 = Linear(hidden_size,
                             hidden_size,
                             weight_init=in_weight_init,
                             bias_init=bias_init,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
        self.dense3.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                          strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                         (parallel_config.model_parallel,)))
        self.dtype = compute_dtype
        self.softmax_dtype = softmax_compute_type
        if self.use_past:
            # operators used for state reuse
            seq_range = np.arange(src_seq_length).reshape(1, 1, -1)
            self.range = Tensor(np.tile(seq_range, (batch_size, 1, 1)), mstype.int32)
            self.seq_length = src_seq_length
            self.attention_mask = Tensor(np.tril(np.ones(shape=(self.seq_length, self.seq_length))), mstype.int32)
            self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
            self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
            self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
            self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
            self.tensor_le = P.LessEqual().shard(((1, 1, 1), (1, 1, 1)))
            self.add = P.Add().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
            self.equal = P.Equal().shard(((1, 1, 1), (1, 1, 1)))
            self.sub1 = P.Sub().shard(((1,), ()))
            self.tile = P.Tile().shard(((1, 1, 1, 1),))
            self.less = P.Less().shard(((1, 1, 1), (1, 1, 1)))
            self.mul1 = P.Mul().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

    def construct(self, query_tensor, key_tensor, value_tensor, alibi_tensor, attention_mask, key_past=None,
                  value_past=None, batch_valid_length=None):
        """Forward process of the MultiHeadAttention"""
        self._check_inputs(query_tensor, key_tensor, value_tensor, attention_mask, key_past,
                           value_past, batch_valid_length)
        ori_shape = F.shape(query_tensor)
        batch_size = self._get_batch_size_from_query(query_tensor)
        query_tensor, key_tensor, value_tensor = self._convert_to_2d_tensor(query_tensor,
                                                                            key_tensor,
                                                                            value_tensor)
        ori_dtype = F.dtype(query_tensor)
        query_tensor = F.cast(query_tensor, self.dtype)
        key_tensor = F.cast(key_tensor, self.dtype)
        value_tensor = F.cast(value_tensor, self.dtype)
        # multi head attention: query, key, value are derived from the same inputs
        query = self.dense1(query_tensor)
        key = self.dense2(key_tensor)
        value = self.dense3(value_tensor)

        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        query = self.transpose(
            F.reshape(
                query,
                (batch_size, self._get_seq_length_under_incremental(self.src_seq_length),
                 self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # the returned shape is [bs, size_per_head, seq_length, num_heads]
        key = self.transpose(
            F.reshape(
                key, (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                      self.n_head, self.size_per_head)),
            (0, 2, 3, 1))
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        value = self.transpose(
            F.reshape(
                value,
                (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                 self.n_head, self.size_per_head)),
            (0, 2, 1, 3))

        # support input shape is [bs, seq, seq] or [bs, heads, seq, seq]
        if attention_mask is not None and len(F.shape(attention_mask)) == 3:
            # expand attention mask from [bs, seq, seq] -> [bs, 1, seq, seq]
            attention_mask = self.expand_dims(attention_mask, 1)
        # key and value for current token(s)
        key_present = key
        value_present = value
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = F.cast(self.less(self.range, batch_valid_length.view(-1, 1, 1)), self.dtype)
                # Cover the key and value numbers corresponding to the padding position
                key_present = self.mul1(key, self.expand_dims(valid_length_vector, 2))
                value_present = self.mul1(value, self.expand_dims(valid_length_vector, 3))
            # The second graph with the inpus size of (bs, 1)
            # the shape of query is (bs, num_heads, 1, size_per_head)
            # the shape of key is   (bs, num_heads, size_per_head, 1)
            # the shape of value is (bs, num_heads, 1, size_per_head)
            else:
                # Get the current token position index
                valid_length = self.reducesum(F.cast(self.not_equal(self.slice(key_past, (0, 0, 0, 0),
                                                                               (F.shape(key_tensor)[0], 1, 1,
                                                                                self.src_seq_length),
                                                                               (1, 1, 1, 1)),
                                                                    0), mstype.float32), (1, 2, 3))
                valid_length = F.reshape(valid_length, (-1, 1, 1))
                valid_length_vector = F.cast(self.equal(valid_length, self.range), self.dtype)
                # Pad the key and value to seq_length with only the position index not zero
                current_key = self.mul1(self.tile(key, (1, 1, 1, self.seq_length)),
                                        self.expand_dims(valid_length_vector, 2))
                current_value = self.mul1(self.tile(value, (1, 1, self.seq_length, 1)),
                                          self.expand_dims(valid_length_vector, 3))
                # Concat the previous saved state and current state
                key = self.add(key_past, current_key)
                value = self.add(value_past, current_value)
                # Update key_present and value_present for state update
                key_present = key
                value_present = value
                attention_mask = F.reshape(self.attention_mask, (self.seq_length, self.seq_length, 1, 1))

        layer_present = (key_present, value_present)
        # multi head attention considering attention mask
        # the return shape is [bs * seq_length, hidden_size]
        attention = self._attn(query, key, value, alibi_tensor, attention_mask)
        # Output
        output = self.projection(attention)
        output = self.dropout(output)
        output = F.reshape(output, ori_shape)
        output = F.cast(output, ori_dtype)
        return output, layer_present

    def _get_batch_size_from_query(self, query):
        r"""Get the batch size from query tensor"""
        # For the incremental prediction, the seq length for the input is 1.
        if len(F.shape(query)) == 2 and ((self.use_past and self.is_first_iteration) or (not self.use_past)):
            return F.shape(query)[0] // self.src_seq_length
        return F.shape(query)[0]

    def _get_seq_length_under_incremental(self, length):
        r"""Return the length of the tensor.
            For the incremental prediction, the seq length for the input is 1.
        """
        if self.use_past and not self.is_first_iteration:
            return 1
        return length

    def _check_inputs(self, query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
                      value_past=None, batch_valid_length=None):
        r"""Check inputs"""
        _check_input_dtype(F.dtype(query_tensor), "query_tensor", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(F.dtype(key_tensor), "key_tensor", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(F.dtype(value_tensor), "value_tensor", [mstype.float32, mstype.float16], self.cls_name)
        if attention_mask is not None:
            _check_input_dtype(F.dtype(attention_mask), "attention_mask", [mstype.float32, mstype.float16],
                               self.cls_name)

        key_is_tensor = isinstance(key_past, Tensor)
        value_is_tensor = isinstance(value_past, Tensor)
        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        key_is_default = key_past is None
        value_is_default = value_past is None
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(self.use_past, "key_past", self.cls_name, None, key_is_tensor,
                                    key_is_default)
        _check_past_none_input_none(self.use_past, "value_past", self.cls_name, None, value_is_tensor,
                                    value_is_default)
        _check_past_none_input_none(self.use_past, "batch_valid_length", self.cls_name, None,
                                    batch_valid_length_is_tensor, batch_is_default)
        if self.use_past:
            _check_input_dtype(F.dtype(key_past), "key_past", [mstype.float16], self.cls_name)
            _check_input_dtype(F.dtype(value_past), "value_past", [mstype.float16], self.cls_name)
            _check_input_dtype(F.dtype(batch_valid_length), "batch_valid_length", [mstype.int32], self.cls_name)
        return True

    def _convert_to_2d_tensor(self, query_tensor, key_tensor, value_tensor):
        """convert a nd tensor to a 2d tensor"""
        query_shape = F.shape(query_tensor)
        query_tensor = F.reshape(query_tensor, (-1, query_shape[-1]))
        key_shape = F.shape(key_tensor)
        key_tensor = F.reshape(key_tensor, (-1, key_shape[-1]))
        value_shape = F.shape(value_tensor)
        value_tensor = F.reshape(value_tensor, (-1, value_shape[-1]))

        return query_tensor, key_tensor, value_tensor

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        x = self.merger_head_transpose(
            x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = P.Shape()(x)
        new_shape = (-1, x_shape[-2] * x_shape[-1])
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _softmax(self, attention_scores):
        """
        For the consideration of the performance, do softmax according to different situations
        :param attention_scores: a 3d tensor before softmax
        :return: the attention scores.
        """

        if self._is_ascend and self.softmax_dtype == mstype.float16 or not self._is_ascend:
            attention_probs = self.softmax(attention_scores)
        else:
            shape = F.shape(attention_scores)
            # attention probs
            attention_probs = self.softmax_3d(
                F.reshape(attention_scores,
                          (shape[0], -1, shape[-1])))
            attention_probs = F.reshape(attention_probs, shape)
        return attention_probs

    def _attn(self, query, key, value, alibi_tensor, attention_mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attention_mask: the attention mask matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        ori_dtype = query.dtype
        score = self.batch_matmul(query.astype(self.dtype), key.astype(self.dtype))
        # score = score.astype(ori_dtype)
        score = self.add_alibi(
            self.mul_alibi(score, self.inv_norm_factor.astype(self.dtype)),
            self.mul_alibi(alibi_tensor, self.beta.astype(self.dtype))
            )
        attention_scores = score.astype(self.softmax_dtype)
        # for input size of (bs, 1) namely the second graph,
        # the shape of attention_mask matrix should be (bs, 1, 1, seq_length)
        if attention_mask is not None:
            if self.use_past and not self.is_first_iteration:
                # Calculate the current total token
                current_index = self.reducesum((self.not_equal(self.slice(key, (0, 0, 0, 0),
                                                                                (query.shape[0], 1, 1,
                                                                                 self.seq_length),
                                                                                (1, 1, 1, 1)),
                                                                     0)).astype(mstype.float32), (1, 2, 3))
                # Get the precise position index
                index = self.sub1(current_index.astype(mstype.int32), 1)
                index = index.reshape((-1, 1, 1))
                # Calculate the attention_mask matrix via the position index
                attention_mask = (self.tensor_le(self.range, index)).astype(mstype.int32)
                attention_mask = self.expand_dims(attention_mask, 2)
            # Minus 10000 for the position where masked to exclude them from softmax
            multiplu_out = self.sub(
                Tensor((1.0,)).astype(attention_scores.dtype),
                attention_mask.astype(attention_scores.dtype))

            adder = self.mul(multiplu_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        # attention probs
        attention_probs = self._softmax(attention_scores)
        attention_probs = attention_probs.astype(ori_dtype)

        attention_probs = self.prob_dropout(attention_probs)
        # Weighted sum output [bs, num_heads, seq_length, size_per_head]

        weighted_values = self.batch_matmul(attention_probs.astype(self.dtype),
                                            value.astype(self.dtype))
        weighted_values = weighted_values.astype(self.softmax_dtype)
        attention_merge = self._merge_heads(weighted_values)
        return attention_merge



# deprecated now
class BloomLayer(Cell):
    """
    pass
    """

    def __init__(self, 
                 config,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 batch_size,
                 seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 use_past=False,
                 in_param_init_func=None,
                 out_param_init_func=None,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 hidden_act='gelu',
                 moe_config=default_moe_config,
                 parallel_config=default_dpmp_config,
                 use_relative_positions=False):
        super().__init__()
        _check_moe_config(moe_config, parallel_config)
        self.use_moe = (moe_config.expert_num > 1)

        # init method, hard code since don't have interface
        in_param_init_func = in_param_init_func if in_param_init_func else 'normal'
        out_param_init_func = out_param_init_func if out_param_init_func else 'normal'
        # import pdb; pdb.set_trace()

        config_to_attention = parallel_config.dpmp if self.use_moe else parallel_config
        _check_config(parallel_config)
        if num_heads % parallel_config.model_parallel != 0:
            raise ValueError("For 'TransformerDecoderLayer', the class variable 'num_heads' must be divisibled by "
                             "'parallel_config.model_parallel', but got the num_heads is {} and "
                             "parallel_config.model_parallel is {}.".format(num_heads,
                                                                            parallel_config.model_parallel))
        if hidden_size % parallel_config.model_parallel != 0:
            raise ValueError(
                "For 'TransformerDecoderLayer', the class variable 'hidden_size' must be divisibled by "
                "'parallel_config.model_parallel', but got the hidden_size is {} and "
                "parallel_config.model_parallel is {}."
                    .format(hidden_size, parallel_config.model_parallel))
        if ffn_hidden_size % parallel_config.model_parallel != 0:
            raise ValueError("For 'TransformerDecoderLayer', the class variable 'ffn_hidden_size' must be "
                             "divisibled by 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
                             "and parallel_config.model_parallel is {}."
                             .format(ffn_hidden_size, parallel_config.model_parallel))
        if use_past:
            raise ValueError(f"The {self.cls_name} does not support use_past=True.")
        self.batch_size = batch_size
        self.use_past = use_past
        self.softmax_compute_type = softmax_compute_type

        self.use_past = use_past
        self.hidden_size = hidden_size

        self.layernorm1 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
        self.layernorm1.shard(((parallel_config.data_parallel, 1),))
        self.layernorm2 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
        self.layernorm2.shard(((parallel_config.data_parallel, 1),))
        self.attention = BloomAttention(config=config,
                                        hidden_size=hidden_size,
                                        num_heads=num_heads,
                                        batch_size=batch_size,
                                        src_seq_length=seq_length,
                                        tgt_seq_length=seq_length,
                                        hidden_dropout_rate=hidden_dropout_rate,
                                        attention_dropout_rate=attention_dropout_rate,
                                        use_past=use_past,
                                        softmax_compute_type=softmax_compute_type,
                                        param_init_type=param_init_type,
                                        parallel_config=config_to_attention,
                                        use_relative_positions=use_relative_positions)

        # Feed Forward Network, FFN
        self.output = BloomFeedForward(config=config,
                                       hidden_size=hidden_size,
                                       dropout_rate=hidden_dropout_rate,
                                       ffn_hidden_size=ffn_hidden_size,
                                       hidden_act=hidden_act,
                                       in_param_init_func=in_param_init_func,
                                       out_param_init_func=out_param_init_func,
                                       param_init_type=param_init_type,
                                       parallel_config=parallel_config)
        self.post_layernorm_residual = post_layernorm_residual
        self.add = P.Add().shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1)))
        self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
        self.dtype = mstype.float16
        self.key_past = None
        self.value_past = None
        if self.use_past:
            # operator used for state reuse
            self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
            self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
            self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
            size_per_head = hidden_size // num_heads
            self.key_shape = (batch_size, num_heads, size_per_head, seq_length)
            self.value_shape = (batch_size, num_heads, seq_length, size_per_head)
            # parameters saving key and value states
            self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
            self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
            self.tile = P.Tile().shard(((1, 1),))
            self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
            self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

    def construct(self, x, alibi_tensor, input_mask=None, init_reset=True, batch_valid_length=None):
        """forward process"""
        if self.post_layernorm_residual:
            input_x = x
        else:
            input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None

        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            self.assign(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
            key_reset = self.key_past
            self.assign(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
            value_reset = self.value_past
            # add dependency for desired execution order
            input_x = F.depend(input_x, key_reset)
            input_x = F.depend(input_x, value_reset)

        attention, layer_present = self.attention(input_x, input_x, input_x, alibi_tensor, input_mask,
                                                  self.key_past, self.value_past, batch_valid_length)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)

        mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            self.assign(self.key_past, key_present)
            key_update = self.key_past
            self.assign(self.value_past, value_present)
            value_update = self.value_past
            # add dependency for desired execution order
            key_update = F.depend(key_update, key_reset)
            value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)

        if self.post_layernorm_residual:
            output = self.add(output_x, mlp_logit)
            output = self.layernorm1(output)
        else:
            output = self.add(x, mlp_logit)

        return output

    def _check_input(self, hidden_states, attention_mask, encoder_output, memory_mask, init_reset, batch_valid_length):
        r"""Check inputs"""
        if not self.use_past or (self.use_past and self.is_first_iteration):
            _check_shape_equal(F.shape(hidden_states), "hidden_states", self.cls_name,
                               [[self.batch_size, self.tgt_seq_length, self.hidden_size],
                                [self.batch_size * self.tgt_seq_length, self.hidden_size]])
            _check_shape_equal(F.shape(attention_mask), "attention_mask", self.cls_name,
                               [self.batch_size, self.tgt_seq_length, self.tgt_seq_length])

        else:
            _check_shape_equal(F.shape(hidden_states), "hidden_states", self.cls_name,
                               [self.batch_size, 1, self.hidden_size])
            _check_shape_equal(F.shape(attention_mask), "attention_mask", self.cls_name,
                               [self.batch_size, 1, self.tgt_seq_length])
        _check_input_dtype(F.dtype(hidden_states), "hidden_states", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(F.dtype(attention_mask), "attention_mask", [mstype.float32, mstype.float16], self.cls_name)
        if encoder_output is not None:
            _check_shape_equal(F.shape(encoder_output), "encoder_output", self.cls_name,
                               [[self.batch_size, self.src_seq_length, self.hidden_size],
                                [self.batch_size * self.src_seq_length, self.hidden_size]])
            _check_input_dtype(F.dtype(encoder_output), "encoder_output",
                               [mstype.float32, mstype.float16], self.cls_name)
        if memory_mask is not None:
            _check_shape_equal(F.shape(memory_mask), "memory_mask", self.cls_name,
                               [self.batch_size, self.tgt_seq_length, self.src_seq_length])
            _check_input_dtype(F.dtype(memory_mask), "memory_mask",
                               [mstype.float32, mstype.float16], self.cls_name)

        init_reset_is_tensor = isinstance(init_reset, Tensor)
        init_reset_is_default = init_reset is True
        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(self.use_past, "init_reset", self.cls_name, True, init_reset_is_tensor,
                                    init_reset_is_default)
        _check_past_none_input_none(self.use_past, "batch_valid_length", self.cls_name, None,
                                    batch_valid_length_is_tensor, batch_is_default)

        if self.use_past:
            _check_shape_equal(F.shape(init_reset), "init_reset", self.cls_name, [1])
            _check_input_dtype(F.dtype(init_reset), "init_reset", [mstype.bool_], self.cls_name)
            _check_shape_equal(F.shape(batch_valid_length), "batch_valid_length", self.cls_name, [self.batch_size])
            _check_input_dtype(F.dtype(batch_valid_length), "batch_valid_length", [mstype.int32], self.cls_name)
        return True
