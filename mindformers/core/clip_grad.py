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
"""Operations for clipping grad by global norm."""

import numpy as np
import hashlib

from mindspore import ops
import mindspore as ms
from mindspore import nn
from mindspore.nn.cell import Cell
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.context import ParallelMode
from mindspore._checkparam import Rel
from mindspore._checkparam import Validator as validator
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_rank, get_group_size, create_group
from mindspore.parallel._ps_context import _is_role_sched
from mindformers.modules.transformer.transformer import default_transformer_config

# The attribute grad_scale is needed for enabling the parallel mode
# If this is removed, c.clip_by_global_norm will have precision error in semi/auto parallel mode.
expand_dims = P.ExpandDims().add_prim_attr("grad_scale", True)
get_square_sum = C.MultitypeFuncGraph("get_square_sum")
apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")
inf = Tensor(np.log(0.0), mstype.float32)
clip_clamped_max = Tensor(1.0, mstype.float32)


@get_square_sum.register("Tensor", "Number")
def _get_square_sum_for_pp(x, value):
    norm = P.ReduceSum(False)(F.square(x), ()) / value
    norm = expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


@get_square_sum.register("Tensor")
def _get_square_sum(x):
    norm = P.ReduceSum(False)(F.square(x), ())
    norm = expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, x):
    x_dtype = F.dtype(x)
    clip_coef = clip_norm / (global_norm + 1e-6)
    # clip_coef_clamped = ops.clip_by_value(clip_coef, clip_value_max=clip_clamped_max, clip_value_min=inf)
    x = x * clip_coef
    x = F.cast(x, x_dtype)
    return x


def _get_model_parallel_group(mp):
    """

    Calculate the communication group of model parallel dim in one pipeline stage

    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    stage_id = rank // per_stage_device_nums
    local_stage_rank_id = rank % per_stage_device_nums
    index = local_stage_rank_id // mp
    group = range(0, mp)
    rank_str_list = [str(x + index * mp + stage_id * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    rank_list = [x + index * mp + stage_id * per_stage_device_nums for x in group]
    return rank_list, rank_list_str


def _get_pipeline_group():
    """

    Calculate the communication group between all pipeline stages

    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    local_stage_rank_id = rank % per_stage_device_nums
    group = range(0, stage_nums)
    rank_list = [local_stage_rank_id + x * per_stage_device_nums for x in group]
    rank_str_list = [str(local_stage_rank_id + x * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    return rank_list, rank_list_str


class GlobalNorm(nn.Cell):
    """
    Calculate the global norm value of given tensors
    """

    def __init__(self, params, parallel_config=default_transformer_config):
        super(GlobalNorm, self).__init__()
        self.parallel_config = parallel_config
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()
        self.is_pipeline = ms.context.get_auto_parallel_context("pipeline_stages") > 1
        self.is_data_parallel = ms.context.get_auto_parallel_context("parallel_mode") == ParallelMode.DATA_PARALLEL
        self.group_size = 1
        if self.is_data_parallel:
            self.merge_op = P.identity()
        else:
            self.merge_op = P.AllReduce()
        if self.is_pipeline:
            pipeline_stages = ms.context.get_auto_parallel_context("pipeline_stages")
            if ms.context.get_auto_parallel_context("enable_parallel_optimizer"):
                self.group_size = get_group_size() // pipeline_stages
            else:
                self.group_size = parallel_config.model_parallel
            group_list, group_name = _get_model_parallel_group(self.group_size)
            # In avoid of the group name too long
            hashed = hashlib.md5(group_name.encode()).hexdigest()[:48]
            print(f"Creating hash value for the group_name hash({group_name})={hashed}")
            group_name = str(hashed)
            create_group(group_name, group_list)
            self.allreduce = P.AllReduce(group=group_name)
            pipeline_group_list, pipeline_group_name = _get_pipeline_group()
            hashed = hashlib.md5(pipeline_group_name.encode()).hexdigest()[:48]
            print(f"Creating hash value for the group_name hash({pipeline_group_name})={hashed}")
            pipeline_group_name = str(hashed)
            create_group(pipeline_group_name, pipeline_group_list)
            self.allreduce2 = P.AllReduce(group=pipeline_group_name)
        else:
            self.group_size = get_group_size()
        self.allreduce_group_size = ()
        if self.is_data_parallel:
            self.allreduce_group_size = (1,) * len(params)
        else:
            self.allreduce_group_size = self._get_scale_for_gradient_norm(params)

    def construct(self, grads):
        """Calculate global norm construct"""
        square_sum = self.hyper_map(get_square_sum, grads, self.allreduce_group_size)
        square_reduce_sum = F.addn(square_sum)
        if self.is_pipeline:
            stage_square_reduce_sum = self.allreduce(square_reduce_sum)
            global_square_reduce_sum = self.allreduce2(stage_square_reduce_sum)
            global_norms = F.sqrt(global_square_reduce_sum)
        else:
            global_norms = F.sqrt(self.merge_op(square_reduce_sum))
        return grads, global_norms

    def _get_scale_for_gradient_norm(self, params):
        allreduce_group_size = ()
        for x in params:
            if "projection.bias" not in x.name and "layernorm" not in x.name and "embedding_table" not in x.name:
                allreduce_group_size = allreduce_group_size + (1.0,)
            elif "embedding_table" not in x.name:
                allreduce_group_size = allreduce_group_size + (self.group_size * 1.0,)
            else:
                if not self.parallel_config.vocab_emb_dp and "position_embedding.embedding_table" not in x.name:
                    allreduce_group_size = allreduce_group_size + \
                                                (self.parallel_config.data_parallel * 1.0,)
                else:
                    allreduce_group_size = allreduce_group_size + (self.group_size * 1.0,)
        return allreduce_group_size


class ClipGradNorm(Cell):
    r"""
    Clips tensor values by the ratio of the sum of their norms.

    Args:
        max_norm (Union(float, int)): The clipping ratio. Default: 1.0
        use_norm (Union(float, None)): The global norm. Default: None

    Inputs:
        - **x** (Union(tuple[Tensor], list[Tensor])) - Input data to clip.

    Outputs:
        Tensor, a clipped Tensor.
    """

    def __init__(self, parameters, max_norm=1.0, use_norm=None, parallel_config=default_transformer_config):
        super(ClipGradNorm, self).__init__()
        # Add interface. This parameter is not used at present
        if use_norm is not None:
            raise ValueError(f"For '{self.cls_name}', input 'use_norm' only supports None currently, "
                             f"but got 'use_norm': {use_norm}")
        validator.check_number("clip_norm", max_norm, 0.0, Rel.GT, self.cls_name)
        self.clip_norm = Tensor([max_norm], mstype.float32)
        self.is_auto_parallel = (ms.context.get_auto_parallel_context("parallel_mode")
                                 in ['semi_auto_parallel', 'auto_parallel'] and not _is_role_sched())
        self.global_norm = None
        if self.is_auto_parallel:
            self.global_norm = GlobalNorm(parameters, parallel_config=parallel_config)
        self.hyper_map = C.HyperMap()
        self.greater_equal = P.GreaterEqual()

    def construct(self, grads):
        """clip grad."""
        if self.global_norm:
            grads, global_norm = self.global_norm(grads)
        else:
            square_sum = self.hyper_map(get_square_sum, grads)
            global_norm = F.sqrt(F.addn(square_sum))
        cond = self.greater_equal(global_norm, self.clip_norm)
        global_norm = F.select(cond, global_norm, self.clip_norm)
        clip_x = self.hyper_map(F.partial(apply_global_norm, self.clip_norm, global_norm), grads)
        return clip_x, global_norm
