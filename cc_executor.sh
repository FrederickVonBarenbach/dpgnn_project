#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=5       # Request GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15      # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=0                 # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-18:00          # DD-HH:MM:SS
#SBATCH --account=def-mlecuyer

module load python/3.7 cuda cudnn

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install --no-index wandb

wandb login 23d9c9f377d31e538d5634ad0280ad712f293a6e
export WANDB_START_METHOD="thread"

# You could also create your environment here, on the local storage ($SLURM_TMPDIR), for better performance. See our docs on virtual environments.

# Prepare data
# mkdir $SLURM_TMPDIR/data
# tar xf ~/projects/def-xxxx/data.tar -C $SLURM_TMPDIR/data

# Start training
echo "Executing on lines $1 to $2"
# check if this exists even $SLURM_GPUS_PER_NODE TODO!!!
awk 'NR >= $1 && NR <= $2' $3 | parallel -j $SLURM_GPUS_PER_NODE --roundrobin 'CUDA_VISIBLE_DEVICES=$(({%} - 1)) python {} &> results/{#}.out'
# python main.py $@