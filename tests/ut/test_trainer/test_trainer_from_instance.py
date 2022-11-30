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
"""
Test module for testing the interface used for mindformers.
How to run this:
python tests/ut/test_trainer/test_trainer_from_instance.py
"""
import numpy as np

from mindspore.nn import AdamWeightDecay, WarmUpLR
from mindspore.train.callback import LossMonitor, TimeMonitor,\
    CheckpointConfig, ModelCheckpoint
from mindspore.dataset import GeneratorDataset

from mindformers.trainer import Trainer
from mindformers.models import MaeModel
from mindformers.common.context import init_context
from mindformers.trainer.config_args import ConfigArguments, \
    RunnerConfig, ContextConfig


class MyDataLoader:
    """Self-Define DataLoader."""
    def __init__(self):
        self._data = [np.zeros((3, 224, 224), np.float32) for _ in range(64)]

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)


def test_trainer_train_from_instance():
    """
    Feature: Create Trainer From Instance
    Description: Test Trainer API to train from self-define instance API.
    Expectation: TypeError
    """
    # example 3: 对任意任务自定义 dataset\model\optimizers\processor\callbacks 实例创建训练
    context_config = ContextConfig(device_id=1, device_target='Ascend', mode=0)
    init_context(seed=2022, use_parallel=False, context_config=context_config)

    # 运行超参配置定义
    runner_config = RunnerConfig(epochs=10, batch_size=8, image_size=224, sink_mode=True, per_epoch_size=10)
    config = ConfigArguments(output_dir="./output_dir", seed=2022, runner_config=runner_config)

    # 自定义模型
    mae_model = MaeModel()

    # 自定义数据集加载及处理流程
    dataset = GeneratorDataset(source=MyDataLoader(), column_names='image')
    dataset = dataset.batch(batch_size=8)

    # 自定义学习策略和优化器
    lr_schedule = WarmUpLR(learning_rate=0.001, warmup_steps=100)
    optimizer = AdamWeightDecay(beta1=0.009, beta2=0.999,
                                learning_rate=lr_schedule,
                                params=mae_model.trainable_params())

    # 自定义callback函数
    loss_cb = LossMonitor(per_print_times=2)
    time_cb = TimeMonitor()
    ckpt_config = CheckpointConfig(save_checkpoint_steps=10, integrated_save=True)
    ckpt_cb = ModelCheckpoint(directory="./output/checkpoint", prefix="my_model", config=ckpt_config)
    callbacks = [loss_cb, time_cb, ckpt_cb]

    mim_trainer_b = Trainer(task_name='masked_image_modeling',
                            model=mae_model,  # 包含loss计算
                            config=config,
                            optimizers=optimizer,
                            train_dataset=dataset,
                            callbacks=callbacks)
    mim_trainer_b.train(resume_from_checkpoint=False)
