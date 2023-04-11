# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Run MindFormer."""
import argparse
import os
from pprint import pprint

import numpy as np

import mindspore as ms
from mindspore.common import set_seed

from mindformers.tools.register import MindFormerConfig, ActionDict
from mindformers.core.parallel_config import build_parallel_config
from mindformers.dataset import check_dataset_config
from mindformers.tools.utils import str2bool
from mindformers.core.context import build_context
from mindformers.trainer import build_trainer
from mindformers.tools.cloud_adapter import cloud_monitor, Obs2Local
from mindformers.tools.logger import logger
from mindformers.mindformer_book import MindFormerBook
from mindspore.parallel._auto_parallel_context import auto_parallel_context

SUPPORT_MODEL_NAMES = MindFormerBook().get_model_name_support_list()

auto_parallel_context().set_enable_all_reduce_fusion(False)
auto_parallel_context().set_enable_all_gather_fusion(False)
auto_parallel_context().set_enable_reduce_scatter_fusion(False)


# @cloud_monitor()
def main(config):
    """main."""
    # init context
    set_seed(config.seed)
    np.random.seed(config.seed)
    if config.parallel_config.pipeline_stage > 1:
        config.parallel.pipeline_stages = config.parallel_config.pipeline_stage

    cfts, profile_cb = build_context(config)

    # build context config
    logger.info(".........Build context config..........")
    build_parallel_config(config)
    logger.info("context config is: %s", config.parallel_config)
    logger.info("moe config is: %s", config.moe_config)

    # # auto pull dataset if on ModelArts platform
    # if config.train_dataset:
    #     config.train_dataset.data_loader.dataset_dir = cfts.get_dataset(
    #         config.train_dataset.data_loader.dataset_dir)
    # if config.eval_dataset:
    #     config.eval_dataset.data_loader.dataset_dir = cfts.get_dataset(
    #         config.eval_dataset.data_loader.dataset_dir)

    if config.run_mode == 'finetune' and not config.resume_or_finetune_checkpoint:
        raise ValueError("if run status is finetune, "
                         "load_checkpoint or resume_or_finetune_checkpoint is invalid, "
                         "it must be input")

    # auto pull checkpoint if on ModelArts platform
    if config.resume_or_finetune_checkpoint:
        # config.resume_or_finetune_checkpoint = cfts.get_checkpoint(config.resume_or_finetune_checkpoint)
        if config.run_mode == 'train':
            config.model.model_config.checkpoint_name_or_path = None
        elif config.run_mode == 'finetune':
            config.model.model_config.checkpoint_name_or_path = config.resume_or_finetune_checkpoint
        else:
            config.model.model_config.checkpoint_name_or_path = config.resume_or_finetune_checkpoint
            config.resume_or_finetune_checkpoint = None

    # define callback and add profile callback
    if config.profile:
        config.profile_cb = profile_cb

    if config.local_rank % 8 == 0:
        pprint(config.parallel_config)
        pprint(config)

    trainer = build_trainer(config.trainer)
    if config.run_mode == 'train' or config.run_mode == 'finetune':
        trainer.train(config, is_full_config=True)
    elif config.run_mode == 'eval':
        trainer.evaluate(config, is_full_config=True)
    elif config.run_mode == 'predict':
        trainer.predict(config, is_full_config=True)


if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default="configs/gpt/run_gpt_2_lm.yaml",
        help='YAML config files')
    parser.add_argument(
        '--mode', default=None, type=int,
        help='Running in GRAPH_MODE(0) or PYNATIVE_MODE(1). Default: GRAPH_MODE(0).'
             'GRAPH_MODE or PYNATIVE_MODE can be set by `mode` attribute and both modes support all backends,'
             'Default: None')
    parser.add_argument(
        '--device_id', default=None, type=int,
        help='ID of the target device, the value must be in [0, device_num_per_host-1], '
             'while device_num_per_host should be no more than 4096. Default: None')
    parser.add_argument(
        '--device_target', default=None, type=str,
        help='The target device to run, support "Ascend", "GPU", and "CPU".'
             'If device target is not set, the version of MindSpore package is used.'
             'Default: None')
    parser.add_argument(
        '--run_mode', default=None, type=str,
        help='task running status, it support [train, finetune, eval, predict].'
             'Default: None')
    parser.add_argument(
        '--dataset', default=None, type=str,
        help='the path of the mindrecord dataset file.'
             'Default: None')
    parser.add_argument(
        '--dataset_list', default=["ftfy_pile_mr2", ], type=list,
        help='the name of mindrecord dataset.'
             'Default: None')
    # ["ftfy_pile_mr2", "ftfy_wudao_mr2", "ftfy_zh_sup_mr2", "ftfy_zh_unsup_mr2"]
    parser.add_argument(
        '--dataset_id', default=None, type=str,
        help='the name of mindrecord dataset.'
             'Default: None')
    parser.add_argument(
        '--dataset_dir', default=None, type=str,
        help='dataset directory of data loader to train/finetune/eval. '
             'Default: None')
    parser.add_argument(
        '--dataset_mem', default=None, type=str,
        help='obs dataset directory of data loader to train/finetune/eval. '
             'Default: None')
    parser.add_argument(
        '--num_samples', default=None, type=int,
        help='dataset sample numbers of data loader to train/finetune/eval. '
             'Default: None')
    parser.add_argument(
        '--predict_data', default=None, type=str,
        help='input data for predict, it support real data path or data directory.'
             'Default: None')
    parser.add_argument(
        '--load_checkpoint', default=None, type=str,
        help="load model checkpoint to train/finetune/eval/predict, "
             "it is also support input model name, such as 'mae_vit_base_p16', "
             "please refer to https://gitee.com/mindspore/transformer#%E4%BB%8B%E7%BB%8D."
             "Default: None")
    parser.add_argument(
        '--batch_size', default=None, type=int,
        help='set training batch size. Default: None')
    parser.add_argument(
        '--seed', default=None, type=int,
        help='global random seed to train/finetune.'
             'Default: None')
    parser.add_argument(
        '--use_parallel', default=None, type=str2bool,
        help='whether use parallel mode. Default: None')
    parser.add_argument(
        '--parallel_mode', default=None, type=int,
        help='set parallel mode. Default: None')
    parser.add_argument(
        '--full_batch', default=None, type=str2bool,
        help='set full batch for semi mode. Default: None')
    parser.add_argument(
        '--data_parallel', default=None, type=int,
        help='set data parallel number. Default: None')
    parser.add_argument(
        '--model_parallel', default=None, type=int,
        help='set model parallel number. Default: None')
    parser.add_argument(
        '--pipeline_parallel', default=None, type=int,
        help='set pipeline parallel number. Default: None')
    parser.add_argument(
        '--micro_size', default=None, type=int,
        help='set micro batch number. Default: None')
    parser.add_argument(
        '--optimizer_parallel', default=None, type=str2bool,
        help='whether use optimizer parallel. Default: None')
    parser.add_argument(
        '--optimizer_shard', default=None, type=str2bool,
        help='whether use optimizer shard for parallel config. Default: None')
    parser.add_argument(
        '--acc_steps', default=None, type=int,
        help='set pipeline parallel number. Default: None')
    parser.add_argument(
        '--recompute', default=None, type=str2bool,
        help='whether use recompute. Default: None')
    parser.add_argument(
        '--profile', default=None, type=str2bool,
        help='whether use profile analysis. Default: None')
    parser.add_argument(
        '--options',
        nargs='+',
        action=ActionDict,
        help='override some settings in the used config, the key-value pair'
             'in xxx=yyy format will be merged into config file')

    args_ = parser.parse_args()
    if args_.config is not None:
        args_.config = os.path.join(work_path, args_.config)
    config_ = MindFormerConfig(args_.config)
    if args_.device_id is not None:
        config_.context.device_id = args_.device_id
    if args_.device_target is not None:
        config_.context.device_target = args_.device_target
    if args_.mode is not None:
        config_.context.mode = args_.mode
    if args_.run_mode is not None:
        config_.run_mode = args_.run_mode
    if args_.seed is not None:
        config_.seed = args_.seed
    if args_.batch_size is not None:
        config_.runner_config.batch_size = args_.batch_size
    if args_.use_parallel is not None:
        config_.use_parallel = args_.use_parallel
    if args_.load_checkpoint is not None:
        if (os.path.isdir(args_.load_checkpoint) and os.listdir(args_.load_checkpoint)) \
                or args_.load_checkpoint.endswith(".ckpt"):
            config_.resume_or_finetune_checkpoint = args_.load_checkpoint
    if args_.parallel_mode is not None:
        config_.parallel.parallel_mode = args_.parallel_mode
    if args_.full_batch is not None:
        config_.parallel.full_batch = args_.full_batch
    if args_.optimizer_parallel is not None:
        config_.parallel.enable_parallel_optimizer = args_.optimizer_parallel
    if args_.data_parallel is not None:
        config_.parallel_config.data_parallel = args_.data_parallel
    if args_.optimizer_shard is not None:
        config_.parallel_config.optimizer_shard = args_.optimizer_shard
    if args_.model_parallel is not None:
        config_.parallel_config.model_parallel = args_.model_parallel
    if args_.pipeline_parallel is not None:
        config_.parallel_config.pipeline_stage = args_.pipeline_parallel
    if args_.micro_size is not None:
        config_.parallel_config.micro_batch_num = args_.micro_size
    if args_.acc_steps is not None:
        config_.runner_wrapper.max_accumulation_step = args_.acc_steps
    if args_.recompute is not None:
        config_.recompute_config.recompute = args_.recompute
    if args_.profile is not None:
        config_.profile = args_.profile
    if args_.options is not None:
        config_.merge_from_dict(args_.options)
    assert config_.run_mode in ['train', 'eval', 'predict', 'finetune'], \
        f"run status must be in {['train', 'eval', 'predict', 'finetune']}, but get {config_.run_mode}"
    if args_.dataset_list:  # 列表
        if args_.dataset_mem:
            rank_id = int(os.getenv("RANK_ID", "0"))
            obs2local = Obs2Local(rank_id)
            args_.dataset_mem = obs2local.obs2local(obs_url=args_.dataset_mem, local_url="/cache/data")
            args_.dataset_dir = [os.path.join(args_.dataset_mem, dataset_name) for dataset_name in args_.dataset_list]
        else:
            if args_.dataset_id:
                args_.dataset_dir = os.path.join(args_.dataset, args_.dataset_id)

    if args_.num_samples is not None:
        config_.train_dataset.data_loader.num_samples = args_.num_samples
    if args_.dataset_dir:
        if config_.run_mode == 'train' or config_.run_mode == 'finetune':
            config_.train_dataset.data_loader.dataset_dir = args_.dataset_dir
        if config_.run_mode == 'eval':
            config_.eval_dataset.data_loader.dataset_dir = args_.dataset_dir
    if config_.run_mode == 'predict':
        if args_.predict_data is None:
            logger.info("dataset by config is used as input_data.")
        elif os.path.isdir(args_.predict_data) and os.path.exists(args_.predict_data):
            predict_data = [os.path.join(root, file)
                            for root, _, file_list in os.walk(os.path.join(args_.predict_data)) for file in file_list
                            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")
                            or file.endswith(".JPEG") or file.endswith("bmp")]
            args_.predict_data = predict_data
        config_.input_data = args_.predict_data
    main(config_)
