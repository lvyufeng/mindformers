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
"""HDFS DataLoader."""
import os
import random
import codecs
from itertools import takewhile, repeat
from typing import Optional, Union, List, Tuple

import pyhdfs

from mindspore.dataset import GeneratorDataset

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class WikitextHDFSDataLoader:
    """Wikitext Dataloader For HDFS."""

    def __new__(cls,
                dataset_dir,
                endswith_words: str = '.tokens',
                local_dir: str = './hdfs_dataset',
                hosts: str = None,
                user_name: str = None,
                online_open: bool = True,
                shuffle: bool = True,
                pull_data_file_number: int = 1, **kwargs):
        r"""
        Wikitext Dataloader API For HDFS.

        Args:
            dataset_dir (str): The directory to wikitext with hdfs.
            endswith_words (str): Retrieves the end character of the desired file name.
            local_dir (str): The directory to wikitext with local server.
            online_open (Optional[bool]): Whether to read data directly from the HDFS.
            hosts (Optional[str]):
                List of NameNode HTTP host: port strings, either as ``list`` or a comma separated string.
                Port defaults to 50070 if left unspecified. Note that in Hadoop 3,
                the default NameNode HTTP port changed to 9870;
                the old default of 50070 is left as-is for backwards compatibility.
            user_name (Optional[str]):
                What Hadoop user to run as.
                Defaults to the ``HADOOP_USER_NAME`` environment variable if present, otherwise ``getpass.getuser()``.
            pull_data_file_number (Optional[int]): Number of files copied from hdfs to the local server at a time.
            shuffle (Optional[bool]): Whether or not to perform shuffle on the dataset.
                Random accessible input is required.
                Default: True, expected order behavior shown in the table below.

        Return:
            A GeneratorDataset for Wikitext dataset on HDFS.

        Raises:
            ValueError: Error input for dataset_dir.
            TypeError: Type error for column_names.

        Examples:
            >>> from mindformers import WikitextHDFSDataLoader
            >>> data_loader = WikitextHDFSDataLoader(dataset_dir="/data/wikitext",
            ...                                      endswith_words=".tokens",
            ...                                      local_dir="/cache/wikitext",
            ...                                      hosts="http://127.0.0.1:50070/",
            ...                                      user_name="root",
            ...                                      shuffle=True,
            ...                                      pull_data_file_number=2,
            ...                                      column_names=["input_ids"])
            >>> data_loader = data_loader.batch(1)
            >>> for item in data_loader:
            >>>     print(item)
            >>>     break

        """
        hdfs_dataloader = HDFSDataset(
            dataset_dir=dataset_dir, endswith_words=endswith_words, hosts=hosts,
            user_name=user_name, online_open=online_open, shuffle=shuffle,
            local_dir=local_dir, pull_data_file_number=pull_data_file_number)
        return GeneratorDataset(source=hdfs_dataloader, shuffle=shuffle, **kwargs)


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class HDFSDataset:
    """HDFS Dataset."""

    def __init__(self,
                 dataset_dir,
                 endswith_words: str = '.tokens',
                 local_dir: str = './hdfs_dataset',
                 hosts: str = None,
                 user_name: str = None,
                 online_open: bool = True,
                 shuffle: bool = True,
                 pull_data_file_number: int = 1):
        r"""
        HDFS Dataset.

        Args:
            dataset_dir (str): The directory to wikitext with hdfs.
            endswith_words (str): Retrieves the end character of the desired file name.
            local_dir (Optional[Union[List[str], Tuple[str]]]): The directory to wikitext with local server.
            online_open (Optional[bool]): Whether to read data directly from the HDFS.
            hosts (Optional[str]):
                List of NameNode HTTP host:port strings, either as ``list`` or a comma separated string.
                Port defaults to 50070 if left unspecified. Note that in Hadoop 3,
                the default NameNode HTTP port changed to 9870;
                the old default of 50070 is left as-is for backwards compatibility.
            user_name (Optional[str]):
                What Hadoop user to run as.
                Defaults to the ``HADOOP_USER_NAME`` environment variable if present, otherwise ``getpass.getuser()``.
            pull_data_file_number (Optional[int]): Number of files copied from hdfs to the local server at a time.

        Return:
            A GeneratorDataset for Wikitext dataset on HDFS.

        Raises:
            ValueError: Error input for dataset_dir.
            TypeError: Type error for column_names.
        """
        self.hosts = hosts
        self.user_name = user_name
        self.local_dir = local_dir
        self.shuffle = shuffle
        self.pull_data_file_number = pull_data_file_number
        self.current_index = self.pull_data_file_number
        self.start_index = 0
        self.hdfs = None
        self.online_open = online_open
        self.local_simulation = False

        if not os.path.isdir(dataset_dir) or not os.path.exists(dataset_dir):
            raise NotADirectoryError(f"dataset_dir is not a directory, it is {dataset_dir}.")

        if local_dir is not None and not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)

        # 本地模拟关闭以下hdfs注册
        if not self.local_simulation:
            self._register_hdfs()
            self._check_hdfs_client()

        self.hdfs_files = self._walk_hdfs_files(dataset_dir, end_words=endswith_words)
        self.local_files = self._convert_local_file_path()
        self.total_samples_number = self._get_hdfs_all_samples_number()

        self.current_local_files = []
        self.current_hdfs_files = []
        self.local_samples_number = 0
        self.local_samples_file = []
        self.hdfs_samples_file = []
        self.local_index = 0
        self.global_count = 0

        if not self.online_open:
            logger.info("When HDFS online reading is disabled, "
                        "%s amount of files endswith %s is pulled in real time for loading",
                        pull_data_file_number, endswith_words)
            self._update_local_files()
        else:
            logger.info("HDFS online reading If this function is enabled, data is read directly from the HDFS")
            self._read_hdfs_files()

    def __getitem__(self, item):
        if not self.online_open:
            if self.global_count == self.total_samples_number:
                self._reset_epoch_index()

            if self.local_index % self.local_samples_number == 0 and self.local_index != 0:
                self._reset_index()

            data_item = self.local_samples_file[self.local_index]

            self.local_index += 1
            self.global_count += 1
        else:
            data_item = self.hdfs_samples_file[item]
        return data_item

    def __len__(self):
        return self.total_samples_number

    def _reset_index(self):
        """Reset index and update files."""
        self.local_samples_file = []
        self.local_index = 0
        self._update_local_files()

    def _reset_epoch_index(self):
        """Reset the index of epoch."""
        self.current_index = self.pull_data_file_number
        self.start_index = 0
        self.global_count = 0

    def _update_local_files(self):
        """Update local file from HDFS."""
        self.current_local_files = self.local_files[self.start_index:self.current_index]
        self.current_hdfs_files = self.hdfs_files[self.start_index:self.current_index]
        self._pull_hdfs_files_to_local()
        self.start_index += self.pull_data_file_number
        if (self.current_index + self.pull_data_file_number) > self.total_samples_number:
            self.current_index = self.total_samples_number - self.current_index
        else:
            self.current_index += self.pull_data_file_number
        self._read_local_files()
        if self.shuffle:
            random.shuffle(self.local_samples_file)

    def _register_hdfs(self, retry: int = 5):
        """Register HDFS."""
        for i in range(retry):
            try:
                self.hdfs = pyhdfs.HdfsClient(
                    hosts=self.hosts, user_name=self.user_name)
            except RuntimeError as error:
                logger.error("%s", error)
                logger.info("HDFS register failed, will retry the %s times.", i)
                continue
            break

    def _read_local_files(self):
        """Get local samples."""
        for local_file in self.current_local_files:
            with codecs.open(local_file, 'r', encoding='utf-8') as file:
                data_items = file.readlines()
                self.local_samples_file.extend([data_item.strip() for data_item in data_items])
        self.local_samples_number = len(self.local_samples_file)

    def _read_hdfs_files(self):
        """Get HDFS samples."""
        for hdfs_file in self.hdfs_files:
            # 模拟HDFS过程
            if self.local_simulation:
                with codecs.open(hdfs_file[0]) as file:
                    data_items = file.readlines()
                    self.hdfs_samples_file.extend([data_item.strip() for data_item in data_items])
            else:
                # 以下注释为HDFS实时读取数据代码
                data_items = self._open_hdfs_file(hdfs_file[0]).readlines()
                self.hdfs_samples_file.extend(data_items)
                # data_items.close()

    def _pull_hdfs_files_to_local(self):
        """Copy HDFS files to local."""
        copy_status = [self._pull_hdfs_file_to_local(
            self.current_hdfs_files[i][0], self.current_local_files[i])
            for i in range(len(self.current_local_files))]
        logger.info("The current number of pull files is %s, with %s successful and %s failed",
                    len(self.current_local_files), sum(copy_status),
                    len(self.current_local_files) - sum(copy_status))

    def _pull_hdfs_file_to_local(self, hdfs_file, local_file, retry: int = 5):
        """Copy HDFS file to local file."""
        for i in range(retry):
            try:
                # 模拟hdfs过程
                if self.local_simulation:
                    import shutil
                    shutil.copy(hdfs_file, local_file)
                else:
                    # 实际hdfs copy代码
                    self.hdfs.copy_to_local(hdfs_path=hdfs_file, local_path=local_file, overwrite=True)
            except RuntimeError as error:
                logger.error("%s", error)
                logger.info("From hdfs_file: %s to local_file: %s failed, will retry the %s times.",
                            hdfs_file, local_file, i)
                continue
            if os.path.exists(local_file):
                return 1
        return 0

    def _open_hdfs_file(self, hdfs_file, retry: int = 5):
        for i in range(retry):
            try:
                hdfs_file = self.hdfs.open(hdfs_file, mode='r', encoding='utf-8')
            except RuntimeError as error:
                logger.error("%s", error)
                logger.info("Open hdfs_file: %s failed, will retry the %s times.",
                            hdfs_file, i)
                continue
            break
        return hdfs_file

    def _walk_hdfs_files(self, dataset_dir, end_words):
        """Walk files ending in end_words."""
        # 实际HDFS遍历代码
        if self.local_simulation:
            return [[os.path.join(root_dir, file), file]
                    for root_dir, _, files in os.walk(dataset_dir) for file in files
                    if file.endswith(end_words)]
        return [[os.path.join(root_dir, file), file]
                for root_dir, _, files in self.hdfs.walk(dataset_dir) for file in files
                if file.endswith(end_words)]

    def _check_hdfs_client(self):
        """Check hdfs client."""
        if self.hdfs is None:
            raise NotImplementedError("HDFS client not register, hdfs is None.")

    def _convert_local_file_path(self):
        """Get local path list."""
        return [os.path.join(self.local_dir, hdfs_file[1]) for hdfs_file in self.hdfs_files]

    def _get_hdfs_all_samples_number(self):
        """Get the number of all hdfs samples."""
        buffer = 1024 * 1024
        num_samples = 0
        for hdfs_file in self.hdfs_files:
            # 模拟HDFS过程
            if self.local_simulation:
                with codecs.open(hdfs_file[0]) as file:
                    buffer_generator = takewhile(lambda x: x, (file.read(buffer)
                                                               for _ in repeat(None)))
                    num_samples += sum(buf.count('\n') for buf in buffer_generator)
            else:
                # 实际HDFS获取样本数量代码
                buffer_generator = takewhile(lambda x: x, (self._open_hdfs_file(hdfs_file[0]).read(buffer)
                                                           for _ in repeat(None)))
                num_samples += sum(buf.count('\n') for buf in buffer_generator)
        return num_samples

    def _get_local_samples_number(self):
        """Get the number of local samples."""
        buffer = 1024 * 1024
        num_samples = 0
        for local_file in self.current_local_files:
            with open(local_file) as file:
                buffer_generator = takewhile(lambda x: x, (file.read(buffer) for _ in repeat(None)))
                num_samples += sum(buf.count('\n') for buf in buffer_generator)
        return num_samples


if __name__ == "__main__":
    hdfs_dir = "./en-wikitext"
    local_dir = "./local_wikitext"
    dataset = WikitextHDFSDataLoader(dataset_dir=hdfs_dir,
                                     local_dir=local_dir,
                                     pull_data_file_number=4,
                                     endswith_words=".tokens",
                                     shuffle=False,
                                     online_open=False,
                                     column_names=["input_ids"])
    dataset.batch(100, drop_remainder=True)

    print(dataset.get_dataset_size())
    dataset_iter = dataset.create_dict_iterator(num_epochs=2)
    for i in range(2):
        print(i)
        for data in dataset_iter:
            pass
