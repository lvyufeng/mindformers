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

from mindspore.common import set_seed

from mindformers.tools.register import MindFormerConfig, ActionDict
from mindformers.common.parallel_config import build_parallel_config
from mindformers.tools.utils import str2bool
from mindformers.common.context import build_context
from mindformers.trainer import build_trainer
from mindformers.tools.cloud_adapter import cloud_monitor
from mindformers.tools.logger import logger
from mindformers.mindformer_book import MindFormerBook


SUPPORT_MODEL_NAMES = MindFormerBook().get_model_name_support_list()


@cloud_monitor()
def main(config):
    """main."""
    # init context
    set_seed(config.seed)
    np.random.seed(config.seed)
    cfts, profile_cb = build_context(config)

    # build context config
    logger.info(".........Build context config..........")
    build_parallel_config(config)
    logger.info("context config is: %s", config.parallel_config)
    logger.info("moe config is: %s", config.moe_config)

    # auto pull dataset if on ModelArts platform
    if config.train_dataset:
        config.train_dataset.data_loader.dataset_dir = cfts.get_dataset(
            config.train_dataset.data_loader.dataset_dir)
    if config.eval_dataset:
        config.eval_dataset.data_loader.dataset_dir = cfts.get_dataset(
            config.eval_dataset.data_loader.dataset_dir)

    # auto pull checkpoint if on ModelArts platform
    if config.resume_checkpoint_path and config.resume_checkpoint_path != '':
        config.resume_checkpoint_path = cfts.get_checkpoint(config.resume_checkpoint_path)
        if isinstance(config.resume_checkpoint_path, str) and config.resume_checkpoint_path in SUPPORT_MODEL_NAMES:
            config.model.model_config.checkpoint_name_or_path = config.resume_checkpoint_path
        else:
            config.model.model_config.checkpoint_name_or_path = None

    # define callback and add profile callback
    if config.profile:
        config.profile_cb = profile_cb

    if config.local_rank % 8 == 0:
        pprint(config)

    trainer = build_trainer(config.trainer)
    if config.run_status == 'train':
        trainer.train(config)
    elif config.run_status == 'eval':
        trainer.evaluate(config)
    elif config.run_status == 'predict':
        trainer.predict(config)


if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=os.path.join(
            work_path, "configs/mae/run_mae_vit_base_p16_224_400ep.yaml"),
        help='YAML config files')
    parser.add_argument('--mode', default=None, type=int, help='context mode')
    parser.add_argument('--device_id', default=None, type=int, help='device id')
    parser.add_argument('--device_target', default=None, type=str, help='device target')
    parser.add_argument('--run_status', default=None, type=str, help='open training')
    parser.add_argument('--dataset_dir', default=None, type=str, help='dataset directory')
    parser.add_argument('--predict_data', default=None, type=str, help='input data for predict')
    parser.add_argument('--resume_checkpoint_path', default=None, type=str, help='load model checkpoint')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--use_parallel', default=None, type=str2bool, help='whether use parallel mode')
    parser.add_argument('--profile', default=None, type=str2bool, help='whether use profile analysis')
    parser.add_argument(
        '--options',
        nargs='+',
        action=ActionDict,
        help='override some settings in the used config, the key-value pair'
             'in xxx=yyy format will be merged into config file')

    args_ = parser.parse_args()
    config_ = MindFormerConfig(args_.config)
    if args_.device_id is not None:
        config_.context.device_id = args_.device_id
    if args_.device_target is not None:
        config_.context.device_target = args_.device_target
    if args_.mode is not None:
        config_.context.mode = args_.mode
    if args_.run_status is not None:
        config_.run_status = args_.run_status
    if args_.seed is not None:
        config_.seed = args_.seed
    if args_.use_parallel is not None:
        config_.use_parallel = args_.use_parallel
    if args_.resume_checkpoint_path is not None:
        config_.resume_checkpoint_path = args_.resume_checkpoint_path
    if args_.profile is not None:
        config_.profile = args_.profile
    if args_.options is not None:
        config_.merge_from_dict(args_.options)
    assert config_.run_status in ['train', 'eval', 'predict'], \
        f"run status must be in {['train', 'eval', 'predict']}, but get {config_.run_status}"
    if args_.dataset_dir:
        if config_.run_status == 'train':
            config_.train_dataset.data_loader.dataset_dir = args_.dataset_dir
        if config_.run_status == 'eval':
            config_.eval_dataset.data_loader.dataset_dir = args_.dataset_dir
    if config_.run_status == 'predict':
        if args_.predict_data is None:
            raise ValueError("predict_data argument must be input, but get None.")
        if os.path.isdir(args_.predict_data) and os.path.exists(args_.predict_data):
            predict_data = [os.path.join(root, file)
                            for root, _, file_list in os.walk(os.path.join(args_.predict_data)) for file in file_list
                            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")
                            or file.endswith(".JPEG") or file.endswith("bmp")]
            args_.predict_data = predict_data
        config_.input_data = args_.predict_data
    main(config_)
