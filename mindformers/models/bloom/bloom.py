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

import mindspore.common.dtype as mstype
import mindspore.nn as nn
import numpy as np
from mindformers.core.loss import CrossEntropyLoss
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.base_model import BaseModel
from mindformers.modules.layers import LayerNorm, Dropout
from mindformers.modules.transformer import AttentionMask, VocabEmbedding, TransformerOpParallelConfig
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from .bloom_config import BloomConfig
from .bloom_modules import BloomLayer, AlibiTensor
from .initializer import SmallNormal, WangNormal

__all__ = ['GPT2LMHeadModel']


default_transformer_config = TransformerOpParallelConfig()




@MindFormerRegister.register(MindFormerModuleType.MODELS)
class BloomLMHeadModel(BaseModel):
    """
            Provide gpt training loss or logits through network.

            Args:
                config (Gpt2Config): The config of Gpt2Model.

            Returns:
                Tensor, the loss or logits of the network.
        """
    _support_list = MindFormerBook.get_model_support_list()['gpt2']

    def __init__(self, config: BloomConfig = None):
        config = config if config is not None else BloomConfig()
        super().__init__(config, auto_prefix=False)

        self.eos_token = self.config.eos_token
        parallel_config = self.config.parallel_config
        self.stridedslice = P.StridedSlice().shard(((parallel_config.data_parallel, 1),))
        self.not_equal = P.NotEqual().shard(((parallel_config.data_parallel, 1), ()))

        self.backbone = BloomModel(self.config)
        self.head = BloomHead(hidden_size=config.embedding_size,
                            vocab_size=config.vocab_size,
                            parallel_config=self.config.parallel_config)
        if parallel_config.pipeline_stage > 1:
            self.head.pipeline_stage = parallel_config.pipeline_stage - 1
            self.backbone.embedding.word_embeddings.embedding_table.add_pipeline_stage(self.head.pipeline_stage)

        mp = config.parallel_config.model_parallel
        vocab_size = config.vocab_size
        if vocab_size % mp != 0:
            logger.warning("The vocab size of GPT Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of GPT Loss will be changed: mp = 1")
            parallel_config.model_parallel = 1

        self.loss = CrossEntropyLoss(parallel_config=parallel_config)
        self.reshape = P.Reshape()
        self.cast = P.Cast()

    def construct(self, input_ids):
        """
        construct function for Language Modeling

        Args:
            input_ids (Tensor): the indices of input sequence tokens in the vocabulary.

        Returns:
            logits (Tensor) or loss (mstype.float32): if is_training is False, directly return the logits,
                                                         otherwise, return the computed loss.
        """

        batch_size, seq_length = input_ids.shape

        tokens = self.stridedslice(input_ids, (0, 0), (batch_size, -1), (1, 1))

        input_mask = self.cast(self.not_equal(tokens, self.eos_token), mstype.float32)

        # [batch_size, seq_length, vocab_size]
        output_states, embedding_table = self.backbone(tokens, input_mask)
        logits = self.head(output_states, embedding_table)

        if self.phase != 'train':
            return logits

        labels = self.stridedslice(input_ids, (0, 1), (batch_size, seq_length), (1, 1))
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss


class BloomEmbeddingLayer(nn.Cell):
    r"""The Embedding Layer of GPT-2 network."""
    def __init__(self, config = None):
        super().__init__(auto_prefix=False)
        parallel_config = copy.deepcopy(config.parallel_config)
        embedding_mp = config.parallel_config.embedding_dp_mp_config.model_parallel
        vocab_size = config.vocab_size
        if vocab_size % embedding_mp != 0:
            logger.warning("The vocab size of embedding layer is: %s, it is not divide by model_parallel: %s",
                           vocab_size, embedding_mp)
            logger.warning("Now, model_parallel will be changed: mp = 1")
            parallel_config.embedding_dp_mp_config.model_parallel = 1

        self.word_embeddings = VocabEmbedding(vocab_size=vocab_size,
                                              embedding_size=config.embedding_size,
                                              param_init=initializer(TruncatedNormal(config.initializer_range),
                                                                    [vocab_size, config.embedding_size],
                                                                    dtype=mstype.float32))

        new_parallel_config = copy.deepcopy(parallel_config)
        new_parallel_config.vocab_emb_dp = True

        self.norm = LayerNorm((config.embedding_size,))

    def construct(self, input_ids):
        """The forward compute of Embedding Layer."""
        word_embedding, word_table = self.word_embeddings(input_ids)
        embedding = self.norm(word_embedding)
        return embedding, word_table


def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.


        Args:
            network(Cell) - Represents the transformer block
            parallel_config(dict) - Parallel Config
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
    """
    # 44layers --> 16stage --> 44//16 == 15 * 2 = 30层(4亿)  12层 --> last_stage （12*2=24亿）
    # Used for the pipeline's stages setting
    # As the final layer is not included here, so we need to manually add here.
    # original:  if set two stages, layers on two stages will be [15, 16+1]
    # with 1 added, the layers on two stages will be [16, 15 +1]
    pp_dis = max(int((layers + 1) / parallel_config.pipeline_stage), 1)
    # the pipeline stage must be in [0, parallel_config.pipeline_stage - 1]
    pp_id = min((layer_id + offset) // pp_dis, parallel_config.pipeline_stage - 1)
    network.pipeline_stage = pp_id
    # print(f"pipeline stage id is {pp_id}", flush=True)

    # Used for optimizer's fusion tag
    dis = max(int((layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        # we give the fusion in pipeline mode a fixed value, otherwise the performance may become worse.
        # network.set_comm_fusion(2)
        # network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
        network.set_comm_fusion(layer_id + 1)
    else:
        network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
    # Used for enabling recomputation of the block
    if isinstance(parallel_config.recompute, bool):
        if parallel_config.recompute:
            network.recompute()
    else:
        if parallel_config.recompute.recompute:
            network.recompute(recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)


def set_parallel_configure_for_layer_20b(network, pp_id, offset, parallel_config, layers, layer_id):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.


        Args:
            network(Cell) - Represents the transformer block
            parallel_config(dict) - Parallel Config
            pp_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
    """
    # 16stage [14 * 3 1 1]
    network.pipeline_stage = pp_id
    # Used for optimizer's fusion tag
    dis = max(int((layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        # we give the fusion in pipeline mode a fixed value, otherwise the performance may become worse.
        network.set_comm_fusion(2)
    else:
        network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
    # Used for enabling recomputation of the block
    if isinstance(parallel_config.recompute, bool):
        if parallel_config.recompute:
            network.recompute()
    else:
        if parallel_config.recompute.recompute:
            network.recompute(recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)


def generate_pp_id_list(layers, per_stage_layers):
    """Generate pipeline stage id list."""
    index_id = 0
    uniform_layer_group = layers // per_stage_layers
    flag_layer = uniform_layer_group * per_stage_layers

    pp_id_list = []

    for i in range(layers):
        if i % per_stage_layers == 0 and index_id < uniform_layer_group:
            pp_id_list.extend([index_id] * per_stage_layers)
            index_id += 1

        if i >= flag_layer:
            pp_id_list.append(index_id)
            index_id += 1
    print("pp_id_list is:{}".format(pp_id_list))
    return pp_id_list


def generate_pp_id_list_new(layers, per_stage_layers):
    """Generate pipeline stage id list."""
    # [1, 14*3, 1] 16stage
    uniform_layer_group = layers // per_stage_layers
    flag_layer = uniform_layer_group * per_stage_layers

    pp_id_list = []
    index_id = 0
    for i in range(layers):
        if i == 0:
            pp_id_list.append(index_id)
            index_id += 1

        if (i - 1)% per_stage_layers == 0 and i < layers - 1:
            pp_id_list.extend([index_id] * per_stage_layers)
            index_id += 1

        if (i - 1) >= flag_layer:
            pp_id_list.append(index_id)
            index_id += 1
    print("pp_id_list is:{}".format(pp_id_list))
    return pp_id_list


class BloomModel(nn.Cell):
    """
    The backbone of GPT network

    Args:
        config(GPT2Config): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32
        input_mask: the mask indicating whether each position is a valid input

    Returns:
        output_state: Tensor, the output logit of backbone
        present_layer: Tensor, the current feature map
        embedding_table: Tensor, the embedding table for the vocabulary
    """

    def __init__(self, config):
        super().__init__()

        self.embedding = BloomEmbeddingLayer(config)
        self.embedding.pipeline_stage = 0

        self.layernorm = LayerNorm((config.embedding_size,)).to_float(config.layernorm_dtype)
        if config.parallel_config.pipeline_stage > 1:
            self.layernorm.set_comm_fusion(40)
        else:
            self.layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        self.layernorm.shard(((config.parallel_config.data_parallel, 1),))
        self.layernorm.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.make_causal_attention = AttentionMask(seq_length=config.seq_length,
                                                parallel_config=config.parallel_config.dp_mp_config)

        self.build_alibi_tensor = AlibiTensor(seq_length=config.seq_length, num_heads=config.num_heads)

        self.num_layers = config.num_layers
        self.blocks = nn.CellList()
        in_param_init_func = SmallNormal(hidden_size=config.embedding_size)
        out_param_init_func = WangNormal(hidden_size=config.embedding_size, n_layers=config.num_layers)
        pp_id_list = generate_pp_id_list_new(layers=self.num_layers, per_stage_layers=config.per_stage_layers)
        for i in range(self.num_layers):
            block = BloomLayer(
                hidden_size=config.embedding_size,
                batch_size=config.batch_size,
                ffn_hidden_size=config.embedding_size * config.expand_ratio,
                seq_length=config.seq_length,
                num_heads=config.num_heads,
                in_param_init_func=in_param_init_func,
                out_param_init_func=out_param_init_func,
                use_relative_positions=config.use_relative_positions,
                attention_dropout_rate=config.attention_probs_dropout_prob,
                hidden_dropout_rate=config.hidden_dropout_prob,
                hidden_act=config.hidden_act,
                param_init_type=config.param_init_type,
                layernorm_compute_type=config.layernorm_dtype,
                softmax_compute_type=config.softmax_dtype,
                parallel_config=config.parallel_config.dp_mp_config)

            set_parallel_configure_for_layer(
                block, layer_id=i, layers=self.num_layers,
                offset=0, parallel_config=config.parallel_config)
            # set_parallel_configure_for_layer_20b(
            #     block, pp_id=pp_id_list[i], layers=self.num_layers,
            #     offset=0, parallel_config=config.parallel_config, layer_id=i)

            self.blocks.append(block)

        self.cast = P.Cast()
        self.dtype = mstype.float16

    def construct(self, input_ids, input_mask):
        """GPT model"""

        input_embedding, embedding_table = self.embedding(input_ids)


        hidden_states = self.cast(input_embedding, self.dtype)
        causal_mask = self.make_causal_attention(input_mask)
        alibi_tensor = self.build_alibi_tensor(input_mask, hidden_states.dtype)

        hidden_shape = F.shape(hidden_states)
        hidden_states = F.reshape(hidden_states, (-1, hidden_shape[-1]))

        for i in range(self.num_layers):
            hidden_states = self.blocks[i](hidden_states, alibi_tensor, causal_mask)

        output_state = self.layernorm(hidden_states)

        return output_state, embedding_table


class BloomHead(nn.Cell):
    """
    Head for GPT to get the logits of each token in the vocab

    Args:
        config(GPTConfig): the config of network

    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary

    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 compute_type=mstype.float16,
                 parallel_config=None):
        super().__init__()
        copied_parallel_config = copy.deepcopy(parallel_config)
        mp = copied_parallel_config.model_parallel
        if vocab_size % mp != 0:
            logger.warning("The vocab size of GPTHead MatMul is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of GPTHead MatMul will be changed: mp = 1")
            copied_parallel_config.model_parallel = 1

        if copied_parallel_config.pipeline_stage > 1:
            copied_parallel_config.vocab_emb_dp = False
        if copied_parallel_config.vocab_emb_dp:
            self.matmul = P.MatMul(transpose_b=True).shard(((copied_parallel_config.data_parallel, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((copied_parallel_config.data_parallel, 1), (
                copied_parallel_config.model_parallel, 1)))
        self.embedding_size = hidden_size
        self.dtype = compute_type
        self.cast = P.Cast()
        self.reshape = P.Reshape()

    def construct(self, state, embedding_table):
        # output logits over vocabulary [bs*seq_length, vocab_size]
        logits = self.matmul(self.cast(state, self.dtype), self.cast(embedding_table, self.dtype))
        return logits


class CrossEntropyCalculationWithMask(nn.Cell):
    """
    Cross Entropy loss
    """

    def __init__(self, is_training=None, num_labels=None):
        super(CrossEntropyCalculationWithMask, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.is_training = is_training
        self.num_labels = num_labels
        self.log_softmax = P.LogSoftmax(axis=-1)

    def construct(self, logits, label_ids, input_mask=None):
        """
        Calculate loss

        Args:
            logits (Tensor): the probability distribution over vocabulary.
            label_ids (Tensor): the indices of input sequence tokens in the vocabulary.
            input_mask (Tensor): input sentences padding mask, where 0 indicates padding position.

        Returns:
            return_value (Tensor, mstype.float32): if is_training is False, directly return the logits, otherwise,
                                                   return the computed loss.
        """

        # logits [batch * (seq_length-1), vocab_size]   label_ids [batch, seq_length-1]
        logits = self.log_softmax(logits)

        if self.is_training:
            label_ids = self.reshape(label_ids, self.last_idx)  # label_ids [batch * (seq_length-1)]
            one_hot_labels = self.onehot(label_ids, self.num_labels, self.on_value,
                                         self.off_value)  # [batch * (seq_length-1), vocab_size]
            per_example_loss = self.neg(
                self.reduce_sum(one_hot_labels * logits, self.last_idx))  # [batch * (seq_length-1)]

            # for PPL calculation in evaluation
            if input_mask is not None:
                input_mask = self.cast(self.reshape(input_mask, self.last_idx),
                                       mstype.float32)  # [batch * (seq_length-1)]

                valid_loss_sum = self.reduce_sum(input_mask * per_example_loss, ())
                valid_element_sum = self.reduce_sum(input_mask, ()) + self.cast(F.tuple_to_array((1e-5,)),
                                                                                mstype.float32)
                loss = valid_loss_sum / valid_element_sum
            else:
                loss = self.reduce_mean(per_example_loss, self.last_idx)  # a number
            return_value = self.cast(loss, mstype.float32)
        else:
            return_value = logits * 1.0  # [batch * (seq_length-1), vocab_size]

        return return_value
