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
"""Causal Image Modeling Dataset."""
import os

import mindspore.common.dtype as mstype
import mindspore.dataset.transforms.c_transforms as C

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.models import build_tokenizer
from .dataloader import build_dataset_loader
from .base_dataset import BaseDataset


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class CausalLanguageModelDataset(BaseDataset):
    """GPT2 pretrain dataset."""

    def __new__(cls, dataset_config: dict = None):
        logger.info("Now Create GPT2 Dataset.")
        rank_id = int(os.getenv("RANK_ID", "0"))
        device_num = int(os.getenv("RANK_SIZE", "1"))
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._check_device_rank_for_parallel(rank_id, device_num)
        dataset_config.rank_id = rank_id
        dataset_config.device_num = device_num
        if dataset_config.data_loader.type != "MindDataset":
            dataset = cls._process_raw_text_data(dataset_config)
        else:
            dataset = cls._process_mindrecord_data(dataset_config)

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                output_columns=dataset_config.input_columns,
                                num_parallel_workers=dataset_config.num_parallel_workers)

        dataset = dataset.project(columns=dataset_config.input_columns)
        dataset = dataset.repeat(dataset_config.repeat)
        type_cast_op = C.TypeCast(mstype.int32)
        for input_arg in dataset_config.input_columns:
            dataset = dataset.map(operations=type_cast_op, input_columns=input_arg)
        return dataset

    @classmethod
    def _prepare_for_model(cls, dataset, dataset_config):
        """Preprocess data for gpt2 model"""
        tokenizer_config = dataset_config.tokenizer
        tokenizer = build_tokenizer(tokenizer_config)
        max_length = tokenizer_config.max_length

        def map_func(input_data):
            input_data = input_data.tolist()
            input_ids = tokenizer(input_data, padding='max_length', max_length=max_length, truncation=True,
                                  add_special_tokens=False)
            return input_ids.get('input_ids')

        dataset = dataset.map(map_func, input_columns=dataset_config.input_columns,
                              output_columns=dataset_config.input_columns)
        return dataset

    @classmethod
    def _process_raw_text_data(cls, dataset_config):
        """Process the text data"""
        dataset_dir = dataset_config.data_loader.pop("dataset_dir")
        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_dir': dataset_dir,
                                                      'num_shards': dataset_config.device_num,
                                                      'shard_id': dataset_config.rank_id})

        dataset = cls._prepare_for_model(dataset, dataset_config)
        return dataset

    @classmethod
    def _process_mindrecord_data(cls, dataset_config):
        """Process the mindrecord data"""
        if "data_files" not in dataset_config.data_loader \
                and dataset_config.data_loader.dataset_dir:
            dataset_files = []
            data_dir = dataset_config.data_loader.dataset_dir
            if os.path.isdir(data_dir):
                for r, _, f in os.walk(data_dir):
                    for file in f:
                        if file.endswith(".mindrecord"):
                            dataset_files.append(os.path.join(r, file))
            else:
                if data_dir.endswith(".mindrecord"):
                    dataset_files.append(data_dir)
        else:
            dataset_files = list(dataset_config.data_loader.dataset_files)
        dataset_config.data_loader.pop("dataset_dir")
        dataset = build_dataset_loader(
            dataset_config.data_loader, default_args={'dataset_files': dataset_files[0],
                                                      'num_shards': dataset_config.device_num,
                                                      'shard_id': dataset_config.rank_id,
                                                      'columns_list': dataset_config.input_columns})
        return dataset
