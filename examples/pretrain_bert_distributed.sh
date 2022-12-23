#!/bin/bash
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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash examples/pretrain_bert_distributed.sh  DEVICE_NUM HOST_FILE DATA_DIR"
echo "for example: examples/pretrain/pretrain_bert_distributed.sh 8 hostfile /path/dataset"
echo "It is better to use absolute path."
echo "=============================================================================================================="

RANK_SIZE=8
HOSTFILE=$1

mpirun --allow-run-as-root -n $RANK_SIZE --hostfile $HOSTFILE \
      --output-filename run_distributed_train_bert \
      -x NCCL_IB_HCA -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_SOCKET_IFNAME -n $RANK_SIZE \
python run_mindformer.py --config ./configs/bert/run_bert_base_uncased.yaml --use_parallel True --run_status train  > distribute_train_gpu_log.txt 2>&1 &

