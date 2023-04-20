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
# hostname 10.78.145.28
export ARNOLD_WORKER_0_HOST=10.78.145.28
# export ARNOLD_ID=0
export ARNOLD_WORKER_0_PORT=11230
export ARNOLD_NUM=8
ulimit -u unlimited
export DEVICE_NUM=$(($ARNOLD_NUM * 8))
export RANK_SIZE=$DEVICE_NUM
# Launch 8 workers.

export MS_SERVER_NUM=0
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_IF_IP=$(hostname -I |awk '{print $1}')
export MS_SCHED_HOST=${ARNOLD_WORKER_0_HOST}  # Scheduler IP address
export MS_SCHED_PORT=${ARNOLD_WORKER_0_PORT}  # Scheduler port
export MS_ROLE=MS_WORKER
export MS_WORKER_NUM=$(($ARNOLD_NUM * 8))
START_ID=$(($ARNOLD_ID * 8))
END_ID=$(($START_ID + 8))
for((i=$START_ID;i<$END_ID;i++));
do
    export MS_NODE_ID=$i
    echo "[INFO] Start worker sched ip: ${MS_SCHED_HOST}, host ip: ${HCCL_IF_IP}, port: ${MS_SCHED_PORT}, " \
         "mode: MS_WORKER, work num: ${MS_WORKER_NUM}, start_id: ${START_ID}, end_id: ${END_ID}, node id: ${i}"
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

if [ $ARNOLD_ID == 0 ]
then
  # Launch 1 scheduler.
  export MS_ROLE=MS_SCHED
  rm -rf ./sched
  mkdir ./sched
  cp ../*.py ./sched
  cp -r ../configs ./sched
  cp -r ../mindformers ./sched
  cd ./sched || exit
  echo "[INFO] Start scheduler sched ip: ${MS_SCHED_HOST}, host ip: ${HCCL_IF_IP}, port: ${MS_SCHED_PORT}, mode: MS_SCHED"
  python3 run_mindformer.py --config=$CONFIG_FILE --use_parallel=True --run_mode=$RUN_STATUS  > sched.log 2>&1  &
  cd ..
fi
