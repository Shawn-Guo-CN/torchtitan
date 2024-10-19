#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# --- This script is optimized for AWS with EFA
# --- adjust NCCL_BUFFSIZE if you encounter memory
# --- constraint issues or to tune for improved performance.
# ---

#SBATCH --job-name=torchtitan_multi_node
#SBATCH --exclusive
#SBATCH --account=ec280

#SBATCH --ntasks=4
#SBATCH --nodes=4

#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

#SBATCH --time=24:00:00
#SBATCH --output=logs/torchtitan_multi_node_%j.out
#SBATCH --error=logs/torchtitan_multi_node_%j.err

cd /work/ec280/ec280/shawn
# load cuda module
echo "Loading cuda module"
module load nvidia/nvhpc/24.5
module load nvidia/cudnn/9.2.0-cuda-12.4
echo ${NVHPC_ROOT}
export CUDA_HOME=${NVHPC_ROOT}/cuda

# activate vitural environment
source /work/ec280/ec280/shawn/vtorchtitan/bin/activate

# NOTE: remember to update the HOME environment variable
export HOME=/work/ec280/ec280/shawn
export HF_HOME=/work/ec280/ec280/shawn/hf_home

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
# Enable for A100
export FI_PROVIDER="efa"
# Ensure that P2P is available
# export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
# optional debug settings
# export NCCL_DEBUG=INFO
# NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
#export TORCH_DIST_INIT_BARRIER=1
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_multi_test.toml"}

dcgmi profile --pause
# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
srun torchrun --nnodes 4 --nproc_per_node 4 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" ./train.py --job.config_file ${CONFIG_FILE}
dcgmi profile --resume
