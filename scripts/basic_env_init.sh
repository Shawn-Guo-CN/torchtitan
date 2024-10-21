set -e

# NOTE: remember to update the HOME environment variable
export HOME=/work/ec280/ec280/shawn
export HF_HOME=/work/ec280/ec280/shawn/hf_home

# load cuda module
module load nvidia/nvhpc/24.5
module load nvidia/cudnn/9.2.0-cuda-12.4
export CUDA_HOME=${NVHPC_ROOT}/cuda # or export CUDA_HOME=${NVHPC_ROOT}

# activate vitural environment
source /work/ec280/ec280/shawn/vtorchtitan/bin/activate

# change to the working directory
cd /work/ec280/ec280/shawn/torchtitan
