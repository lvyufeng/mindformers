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
"""Trainer Utils."""
import os
import random
from enum import Enum

import numpy as np

from mindspore import context, load_checkpoint, load_param_into_net
from mindspore import set_seed as ms_set_seed

from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerConfig


class BaseEnum(str, Enum):
    """
    Base Enum for MindFormers.
    """

    @classmethod
    def _missing_(cls, value):
        """Enum with more explicit error message for missing values."""
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class SaveIntervalStrategy(BaseEnum):
    """
    Stores the acceptable string identifiers for save checkpoint monitor.
    """
    NO = "no"
    STEPS = "steps"
    SECONDS = "seconds"


class LRType(BaseEnum):
    """
    Stores the acceptable string identifiers for learning rate schedule.
    """
    # supported item for test, will be delete in the future.
    WARMUPCOSINEV1 = "WarmUpCosineDecayV1"

    # will be support item for future.
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class OptimizerType(BaseEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """
    # supported item for test, will be delete in the future.
    ADAMWEIGHTDECAY = 'AdamWeightDecay'

    # will be support item for future.
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAFACTOR = "adafactor"


class WrapperType(BaseEnum):
    """
    Stores the acceptable string identifiers for training wrapper.
    """
    # will be support item for future.
    MFWRAPPER = 'mf_wrapper'
    TRAINONESTEP = 'wrapper'
    TRAINONESTEPWITHLOSSSCALE = 'loss_scale_wrapper'


def set_seed(seed: int = 0):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `MindSpore`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    ms_set_seed(seed)


def check_keywords_in_name(name, keywords=()):
    """ Check keywords in name. """
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def check_runner_config(config, dataset):
    """ Check runner config. """
    data_size = dataset.get_dataset_size()
    new_epochs = config.runner_config.epochs
    config.runner_config.origin_epochs = new_epochs
    if config.runner_config.sink_mode:
        if config.runner_config.per_epoch_size != -1:
            if config.runner_config.per_epoch_size <= 0:
                raise ValueError("per epoch size must be more than 0 or equal to -1, "
                                 f"but get {config.runner_config.per_epoch_size}")
            if data_size < config.runner_config.per_epoch_size:
                logger.warning("The data size %s (get from dataset.get_dataset_size()) is smaller "
                               "than the per_epoch_size %s (get from config.runner_config.per_epoch_size), "
                               "you should set the config.runner_config.per_epoch_size to %s",
                               data_size, config.runner_config.per_epoch_size, data_size)
            config.runner_config.epochs = int((data_size / config.runner_config.per_epoch_size) * new_epochs)
        else:
            config.runner_config.per_epoch_size = data_size
    else:
        logger.warning("Sink mode is False, per epoch size is invalid, it will reset -1.")
        config.runner_config.per_epoch_size = -1

    config.data_size = data_size
    logger.info("Will be Training epochs:%d, sink_size:%d",
                config.runner_config.epochs, config.runner_config.per_epoch_size)
    logger.info("Create training dataset finish, dataset size:%d", data_size)


def check_train_data_loader_type(new_config, old_config):
    """Check train data loader config type."""
    if new_config.train_dataset is None:
        return None
    if new_config.train_dataset.get('data_loader') is None:
        return None
    train_data_loader_type = new_config.train_dataset.get('data_loader').get('type')
    if old_config.train_dataset is not None and train_data_loader_type is not None:
        default_train_data_loader_type = old_config.train_dataset.data_loader.type
        if train_data_loader_type != default_train_data_loader_type:
            logger.warning("train dataset's data_loader type is changed to %s."
                           "The default parameters will be cleared."
                           "Please make sure to input the corresponding parameter values manually.",
                           train_data_loader_type)
            old_config.train_dataset.data_loader = {}
    return None


def check_eval_data_loader_type(new_config, old_config):
    """Check eval data loader config type."""
    if new_config.eval_dataset is None:
        return None
    if new_config.eval_dataset.get('data_loader') is None:
        return None
    eval_data_loader_type = new_config.eval_dataset.get('data_loader').get('type')
    if old_config.eval_dataset is not None and eval_data_loader_type is not None:
        default_eval_data_loader_type = old_config.eval_dataset.data_loader.type
        if eval_data_loader_type != default_eval_data_loader_type:
            logger.warning("eval dataset's data_loader type is changed to %s."
                           "The default parameters will be cleared."
                           "Please make sure to input the corresponding parameter values manually.",
                           eval_data_loader_type)
            old_config.eval_dataset.data_loader = {}
    return None


def check_optimizer_and_lr_type(new_config, old_config):
    """Check optimizer and lr schedule config type."""
    optimizer_type = new_config.optimizer.get('type')
    if old_config.optimizer is not None and optimizer_type is not None:
        default_optimizer_type = old_config.optimizer.type
        if optimizer_type != default_optimizer_type:
            logger.warning(
                "optimizer type is changed to %s."
                "The default parameters will be cleared."
                "Please make sure to input the corresponding parameter values manually except (params).",
                optimizer_type)
            old_config.optimizer = {}

    if hasattr(new_config.optimizer, 'learning_rate'):
        lr_type = new_config.optimizer.learning_rate.get('type')
        if old_config.lr_schedule is not None and lr_type is not None:
            default_lr_type = old_config.lr_schedule.type
            if lr_type != default_lr_type:
                logger.warning(
                    "lr schedule type is changed to %s."
                    "The default parameters will be cleared."
                    "Please make sure to input the corresponding parameter values manually.",
                    lr_type)
                old_config.lr_schedule = None


def check_lr_config(new_config, old_config):
    """Check lr schedule config."""
    lr_type = new_config.lr_schedule.type
    if old_config.lr_schedule is not None and lr_type is not None:
        default_lr_type = old_config.lr_schedule.type
        if lr_type != default_lr_type:
            logger.warning(
                "lr schedule type is changed to %s."
                "The default parameters will be cleared."
                "Please make sure to input the corresponding parameter values manually.",
                lr_type)
            old_config.lr_schedule = None


def _check_lr_config(config, device_num=1, batch_size=128, arch="SwinForImageClassification"):
    if arch in ('SwinForMaskedImageModeling', 'SwinForImageClassification'):
        config.base_lr = (config.base_lr * device_num * batch_size) / 512
        config.min_lr = (config.min_lr * device_num * batch_size) / 512
        config.warmup_lr = (config.warmup_lr * device_num * batch_size) / 512
    if arch in ('ViTMAEForPreTraining', 'ViTForImageClassification'):
        config.base_lr = (config.base_lr * device_num * batch_size) / 256


def check_image_lr_config(config):
    """config lr"""
    lr_config = config.lr_schedule
    device_num = config.device_num
    batch_size = config.runner_config.batch_size
    _check_lr_config(lr_config, device_num=device_num, batch_size=batch_size, arch=config.model.arch.type)
    total_epochs = config.runner_config.epochs
    steps_per_epoch = config.data_size
    if config.runner_config.per_epoch_size != -1:
        total_steps = total_epochs * config.runner_config.per_epoch_size
    else:
        total_steps = total_epochs * steps_per_epoch
    lr_config.warmup_steps = int(lr_config.warmup_epochs * steps_per_epoch)
    if config.lr_schedule.type == 'WarmUpCosineDecayV1':
        lr_config.decay_steps = total_steps - lr_config.warmup_steps
    elif config.lr_schedule.type == 'WarmUpCosineDecayV2':
        lr_config.total_steps = total_steps
    del lr_config.warmup_epochs


def check_wrapper_config(new_config, old_config):
    """Check wrapper config."""
    wrapper_type = new_config.runner_wrapper.get('type')
    if old_config.runner_wrapper is not None and wrapper_type is not None:
        default_wrapper_type = old_config.runner_wrapper.type
        if wrapper_type != default_wrapper_type:
            logger.warning(
                "wrapper type is changed to %s."
                "The default parameters will be cleared."
                "Please make sure to input the corresponding parameter values manually.",
                wrapper_type)
            old_config.runner_wrapper = {}


def config2dict(config):
    """MindFormerConfig Type Convert to Dict."""
    if not isinstance(config, (dict, MindFormerConfig)):
        return config
    new_dict = {}
    for key, value in config.items():
        if isinstance(value, MindFormerConfig):
            value = config2dict(value)
        new_dict.setdefault(key, value)
    return new_dict


def load_distributed_checkpoint():
    """Load Checkpoint in Parallel Mode."""


def resume_checkpoint_for_training(config, network, optimizer):
    """Resume Checkpoint for training."""
    if not os.path.realpath(config.resume_or_finetune_checkpoint) or \
            not os.path.exists(config.resume_or_finetune_checkpoint):
        raise FileNotFoundError(f"The resume_or_finetune_checkpoint must be correct, "
                                f"but get {config.resume_or_finetune_checkpoint}")
    if context.get_auto_parallel_context('parallel_mode') in \
            ['semi_auto_parallel', 'auto_parallel', 'hybrid_parallel']:
        load_distributed_checkpoint()
    else:
        checkpoint_dict = load_checkpoint(config.resume_or_finetune_checkpoint)
        not_load_network_params = load_param_into_net(network, checkpoint_dict)
        not_load_optim_params = load_param_into_net(optimizer, checkpoint_dict)
        logger.info("Not load network parameters is：%s", str(not_load_network_params))
        logger.info("Not load optimizer parameters is：%s", str(not_load_optim_params))


def load_distributed_checkpoint_v2(config, model, dataset):
    """Load Checkpoint in Parallel Mode."""
    if not os.path.isdir(config.resume_or_finetune_checkpoint):
        raise NotADirectoryError(
            "When distributed loads are sliced weights,"
            "resume_or_finetune_checkpoint should be a checkpoint directory containing the directory of rank_{0-*},"
            "The directory structure is as follows: **checkpoint_root_dir/rank_{0-*}/checkpoint/**.ckpt")
    distribute_checkpoint_dir = os.path.join(
        config.resume_or_finetune_checkpoint,
        "rank_{}".format(int(os.getenv("RANK_ID", "0"))), "checkpoint")
    distribute_checkpoint_path = get_last_checkpoint(distribute_checkpoint_dir)
    model.build(train_dataset=dataset, epoch=config.runner_config.epochs)
    checkpoint_dict = load_checkpoint(distribute_checkpoint_path)
    return checkpoint_dict


def resume_checkpoint_for_training_v2(config, model, network, optimizer, dataset):
    """Resume Checkpoint for training."""
    if not os.path.realpath(config.resume_or_finetune_checkpoint) or \
            not os.path.exists(config.resume_or_finetune_checkpoint):
        raise FileNotFoundError(f"The resume_or_finetune_checkpoint must be correct, "
                                f"but get {config.resume_or_finetune_checkpoint}")
    if context.get_auto_parallel_context('parallel_mode') in \
            ['semi_auto_parallel', 'auto_parallel', 'hybrid_parallel']:
        checkpoint_dict = load_distributed_checkpoint_v2(config, model, dataset)
    else:
        checkpoint_dict = load_checkpoint(config.resume_or_finetune_checkpoint)
    not_load_network_params = load_param_into_net(network, checkpoint_dict)
    not_load_optim_params = load_param_into_net(optimizer, checkpoint_dict)
    logger.info("Network parameters are not loaded：%s", str(not_load_network_params))
    logger.info("Optimizer parameters are not loaded：%s", str(not_load_optim_params))


def get_last_checkpoint(checkpoint_dir):
    """get last checkpoint for resuming or finetune."""
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(
            "When distributed loads are sliced weights,"
            "resume_or_finetune_checkpoint should be a checkpoint directory containing the directory of rank_{0-*},"
            "The directory structure is as follows: **checkpoint_root_dir/rank_{0-*}/checkpoint/**.ckpt")
    output_checkpoint_path = [
        checkpoint for checkpoint in os.listdir(checkpoint_dir)
        if checkpoint.endswith('.ckpt')
    ]
    if not output_checkpoint_path:
        return None
    output_checkpoint_path = sorted(output_checkpoint_path,
                                    key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, output_checkpoint_path[-1])
