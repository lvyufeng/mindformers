#!/bin/bash
CURRENT_DIR=$(cd `dirname $0`; pwd)
CURRENT_NAME="${CURRENT_DIR##*/}"

# Step1, download obsutil tools
# wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_arm64.tar.gz /root/
# cd /root/
# tar -xzvf obsutil_linux_arm64.tar.gz
# cd obsutil_linux_arm64_5.3.4/
# chmod 755 obsutil

rm -rf scripts/mf_parallel* || exit

cd /home/linbert/obsutil_linux_arm64_5.3.4/
./obsutil config -i=0PJLSOGVHGX4RVPNXYSG -k=kOLafto3hkGCbrUdCBQRnup6UvHDWJDnU26yeD81 -e=obs.cn-south-222.ai.pcl.cn

#./obsutil ls -s
# obs路径请根据自己的工作目录替换填写
OBS_PATH=obs://bos/idea_gpt
./obsutil sync $CURRENT_DIR $OBS_PATH/$CURRENT_NAME

#./obsutil sync obs://bos/idea_gpt/output/hql666/profile/rank_1 $CURRENT_DIR/profile/rank_1

