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
"""
Test module for testing the gpt interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_gpt_model/test_gpt2_model.py
"""
from dataclasses import dataclass
import os
import numpy as np
import pytest
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.dataset import GeneratorDataset

from mindformers.trainer import Trainer
from mindformers.trainer.config_args import ConfigArguments, \
    RunnerConfig
from mindformers.models.bloom import BloomLMHeadModel, BloomConfig
from mindformers.core.lr import WarmUpDecayLR
from mindformers.core.optim import FusedAdamWeightDecay


def generator():
    """dataset generator"""
    seq_len = 21
    input_ids = np.random.randint(low=0, high=15, size=(seq_len,)).astype(np.int32)
    input_mask = np.ones_like(input_ids)
    label_ids = input_ids
    train_data = (input_ids, input_mask, label_ids)
    for _ in range(512):
        yield train_data[0]

@dataclass
class Tempconfig:
    seed: int = 0
    runner_config: RunnerConfig = None
    data_size: int = 0
    resume_or_finetune_checkpoint: str = ""

@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_bloom_trainer_train_from_instance():
    """
    Feature: Create Trainer From Instance
    Description: Test Trainer API to train from self-define instance API.
    Expectation: TypeError
    """
    # Config definition
    runner_config = RunnerConfig(epochs=1, batch_size=8, sink_mode=True, per_epoch_size=2)
    config = ConfigArguments(seed=2022, runner_config=runner_config)

    # Model
    config = BloomConfig(seq_length=20, vocab_size=2000, num_heads=4, num_layers=2)
    gpt_model = BloomLMHeadModel(config)

    # Dataset and operations
    dataset = GeneratorDataset(generator, column_names=["input_ids"])
    dataset = dataset.batch(batch_size=8)

    # optimizer
    lr_schedule = WarmUpDecayLR(learning_rate=0.0001, end_learning_rate=0.00001, warmup_steps=0, decay_steps=512)
    optimizer = FusedAdamWeightDecay(beta1=0.009, beta2=0.999,
                                     learning_rate=lr_schedule,
                                     params=gpt_model.trainable_params())

    # callback
    loss_cb = LossMonitor(per_print_times=2)
    time_cb = TimeMonitor()
    callbacks = [loss_cb, time_cb]
    print(config)

    lm_trainer = Trainer(model=gpt_model,
                         config=config,
                         optimizers=optimizer,
                         train_dataset=dataset,
                         callbacks=callbacks)
    lm_trainer.train(resume_or_finetune_from_checkpoint=False)
