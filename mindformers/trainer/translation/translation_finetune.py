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
"""Translation Modeling Trainer."""
import os.path
from typing import Optional, List, Union

from mindspore.train import Callback
from mindspore.nn import TrainOneStepCell, Optimizer, Cell
from mindspore.dataset import GeneratorDataset

from mindformers.dataset import BaseDataset
from mindformers.models import build_model, BaseModel, BaseTokenizer
from mindformers.tools.logger import logger
from mindformers.tools.utils import count_params
from mindformers.tools.register import MindFormerRegister,\
    MindFormerModuleType, MindFormerConfig
from mindformers.pipeline import pipeline
from ..base_trainer import BaseTrainer
from ..config_args import ConfigArguments
from ..training_args import TrainingArguments


@MindFormerRegister.register(MindFormerModuleType.TRAINER)
class TranslationTrainer(BaseTrainer):
    r"""Translation Task For Trainer.
    Args:
        model_name (str): The model name of Task-Trainer. Default: None
    Examples:
        >>> from mindformers.trainer import TranslationTrainer
        >>> from mindformers import T5ForConditionalGeneration, TranslationTrainer
        >>> # follow the instruction in t5 section in the README.md and download wmt16 dataset.
        >>> # change the dataset_files path of configs/t5/wmt16_dataset.yaml
        >>> trans_trainer = TranslationTrainer(model_name='t5_small')
        >>> trans_trainer.train()
        >>> model = T5ForConditionalGeneration.from_pretrained('t5_small')
        >>> trans_trainer = TranslationTrainer(model_name="t5_small")
        >>> res = trans_trainer.predict(input_data="hello world", network=model)
            [{'translation_text': ['hello world']}]
    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """
    def __init__(self, model_name: str = None):
        super(TranslationTrainer, self).__init__("translation", model_name)

    def train(self,
              config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
              network: Optional[Union[Cell, BaseModel]] = None,
              dataset: Optional[Union[BaseDataset, GeneratorDataset]] = None,
              wrapper: Optional[TrainOneStepCell] = None,
              optimizer: Optional[Optimizer] = None,
              callbacks: Optional[Union[Callback, List[Callback]]] = None,
              **kwargs):
        r"""Train task for Translation Trainer.
        This function is used to train or fine-tune the network.

        The trainer interface is used to quickly start training for general task.
        It also allows users to customize the network, optimizer, dataset, wrapper, callback.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            network (Optional[Union[Cell, BaseModel]]): The network for trainer.
                It supports model name or BaseModel or MindSpore Cell class.
                Default: None.
            dataset (Optional[Union[BaseDataset, GeneratorDataset]]): The training dataset.
                It support real dataset path or BaseDateset class or MindSpore Dataset class.
                Default: None.
            optimizer (Optional[Optimizer]): The training network's optimizer. It support Optimizer class of MindSpore.
                Default: None.
            wrapper (Optional[TrainOneStepCell]): Wraps the `network` with the `optimizer`.
                It support TrainOneStepCell class of MindSpore.
                Default: None.
            callbacks (Optional[Union[Callback, List[Callback]]]): The training callback function.
                It support CallBack or CallBack List of MindSpore.
                Default: None.
        Raises:
            NotImplementedError: If wrapper not implemented.
        """
        self.training_process(
            config=config,
            network=network,
            callbacks=callbacks,
            dataset=dataset,
            wrapper=wrapper,
            optimizer=optimizer,
            **kwargs)

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError(
            "The Translation task does not support evaluate.")

    def predict(self,
                config: Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]] = None,
                input_data: Optional[Union[str, list, GeneratorDataset]] = None,
                network: Optional[Union[Cell, BaseModel]] = None,
                tokenizer: Optional[BaseTokenizer] = None,
                **kwargs):
        """
        Executes the predict of the trainer.

        Args:
            config (Optional[Union[dict, MindFormerConfig, ConfigArguments, TrainingArguments]]):
                The task config which is used to configure the dataset, the hyper-parameter, optimizer, etc.
                It supports config dict or MindFormerConfig or TrainingArguments or ConfigArguments class.
                Default: None.
            input_data (Optional[Union[Tensor, str, list]]): The predict data. It supports 1) a text string to be
                translated, 1) a file name where each line is a text to be translated  and 3) a generator dataset.
                Default: None.
            network (Optional[Union[Cell, BaseModel]]): The network for trainer.
                It supports model name or BaseModel or MindSpore Cell class.
                Default: None.
            tokenizer (Optional[BaseTokenizer]): The tokenizer for tokenizing the input text.
                Default: None.
        Returns:
            A list of prediction.

        """
        config = self.set_config(config)

        if input_data is None:
            input_data = config.input_data

        if not isinstance(input_data, (str, list, GeneratorDataset)):
            raise ValueError("Input data's type must be one of "
                             f"[str, list, GeneratorDataset], but got type {type(input_data)}")

        logger.info(".........Build Net..........")
        if network is None:
            network = build_model(config.model)

        if network is not None:
            logger.info("Network Parameters: %s M.", str(count_params(network)))

        save_file = kwargs.pop("save_file", None)
        if save_file is None:
            if config and config.save_file is not None:
                save_file = config.save_file
            else:
                save_file = "results.txt"

        pipeline_task = pipeline(task='translation',
                                 tokenizer=tokenizer,
                                 model=network, **kwargs)
        if isinstance(input_data, str) and os.path.isfile(input_data):
            with open(input_data, 'r') as fp:
                output_result = []
                for line in fp:
                    output = pipeline_task(line, **kwargs)
                    output_result.extend(output)
        else:
            output_result = pipeline_task(input_data, **kwargs)

        logger.info(".........start to write the output result to: %s.........", save_file)
        with open(save_file, 'w') as file:
            if isinstance(output_result, list):
                for item in output_result:
                    file.write(str(item) + '\n')
            else:
                file.write(str(output_result))
            file.close()
        logger.info(".........writing result finished..........")
        logger.info(".........Predict Over!.............")
        return output_result
