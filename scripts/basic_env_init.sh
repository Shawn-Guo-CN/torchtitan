set -e

# NOTE: remember to update the HOME environment variable
export HOME=/work/ec280/ec280/shawn
echo "HOME is set to $HOME"
export HF_HOME=/work/ec280/ec280/shawn/hf_home
echo "HF_HOME is set to $HF_HOME"

# load cuda module
module load nvidia/nvhpc/24.5
module load nvidia/cudnn/9.2.0-cuda-12.4
export CUDA_HOME=${NVHPC_ROOT}/cuda # or export CUDA_HOME=${NVHPC_ROOT}
echo "CUDA_HOME is set to $CUDA_HOME"

# activate vitural environment
source /work/ec280/ec280/shawn/vtorchtitan/bin/activate
echo "Virtual environment is activated"

# change to the working directory
cd /work/ec280/ec280/shawn/torchtitan
echo "Current working directory is $(pwd)"