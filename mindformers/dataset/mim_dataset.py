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
"""Masked Image Modeling Dataset."""
import os

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from .dataloader import build_dataset_loader
from .mask import build_mask
from .transforms import build_transforms
from .sampler import build_sampler
from .base_dataset import BaseDataset


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class MIMDataset(BaseDataset):
    """
    Masked Image Modeling Dataset.

    Examples:
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers.dataset import build_dataset, check_dataset_config
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig("configs/mae/run_mae_vit_base_p16_224_800ep.yaml")
        >>> check_dataset_config(config)
        >>> # 1) use config dict to build dataset
        >>> dataset_from_config = build_dataset(config.train_dataset_task)
        >>> # 2) use class name to build dataset
        >>> dataset_from_name = build_dataset(class_name='MIMDataset', dataset_config=config.train_dataset)
        >>> # 3) use class to build dataset
        >>> dataset_from_class = MIMDataset(config.train_dataset)
    """
    def __new__(cls, dataset_config: dict = None):
        logger.info("Now Create Masked Image Modeling Dataset.")
        cls.init_dataset_config(dataset_config)
        rank_id = int(os.getenv("RANK_ID", "0"))
        device_num = int(os.getenv("RANK_SIZE", "1"))

        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'num_shards': device_num, 'shard_id': rank_id})
        transforms = build_transforms(dataset_config.transforms)
        mask = build_mask(dataset_config.mask_policy)
        sampler = build_sampler(dataset_config.sampler)

        if sampler is not None:
            dataset = dataset.use_sampler(sampler)

        if transforms is not None:
            for column in dataset_config.input_columns:
                dataset = dataset.map(
                    input_columns=column,
                    operations=transforms,
                    num_parallel_workers=dataset_config.num_parallel_workers,
                    python_multiprocessing=dataset_config.python_multiprocessing)

        if mask is not None:
            dataset = dataset.map(
                operations=mask,
                input_columns=dataset_config.input_columns,
                column_order=dataset_config.column_order,
                output_columns=dataset_config.output_columns,
                num_parallel_workers=dataset_config.num_parallel_workers,
                python_multiprocessing=dataset_config.python_multiprocessing)
        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                column_order=dataset_config.column_order,
                                output_columns=dataset_config.output_columns,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset
