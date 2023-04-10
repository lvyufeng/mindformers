#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

if [ $# != 2 ]
then
  echo "Usage Help: bash run_distribute.sh [CONFIG_PATH] [RUN_STATUS]"
  exit 1
fi

check_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CONFIG_FILE=$(check_real_path $1)
RUN_STATUS=$2

if [ ! -f $CONFIG_FILE ]
then
    echo "error: config_path=$CONFIG_FILE is not a file"
exit 1
fi

export GLOG_v=2
unset GLOG_log_dir
unset GLOG_logtostderr
ulimit -u unlimited
export DEVICE_NUM=16
export RANK_SIZE=$DEVICE_NUM

export MS_SERVER_NUM=0
export MS_WORKER_NUM=16
export HCCL_IF_IP=10.147.182.217
export MS_SCHED_HOST=127.0.0.1  # Scheduler IP address
export MS_SCHED_PORT=8081             # Scheduler port
export MS_ROLE=MS_WORKER

# Launch 8 workers.
for((i=8;i<16;i++));
do
    export MS_NODE_ID=$i
    rm -rf ./worker_$i
    mkdir ./worker_$i
    cp ../*.py ./worker_$i
    cp -r ../configs ./worker_$i
    cp -r ../mindformers ./worker_$i
    cd ./worker_$i || exit
    env > env.log
    python3 run_mindformer.py --config=$CONFIG_FILE --use_parallel=True --run_mode=$RUN_STATUS  > work.log 2>&1 &
    cd ..
done
