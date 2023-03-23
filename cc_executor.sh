#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=5       # Request GPU
#SBATCH --cpus-per-task=15      # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=0                 # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-18:00          # DD-HH:MM:SS
#SBATCH --account=def-mlecuyer

module load python/3.9 cuda cudnn

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

wandb login 23d9c9f377d31e538d5634ad0280ad712f293a6e
export WANDB_START_METHOD="thread"

# You could also create your environment here, on the local storage ($SLURM_TMPDIR), for better performance. See our docs on virtual environments.

# Prepare data
# mkdir $SLURM_TMPDIR/data
# tar xf ~/projects/def-xxxx/data.tar -C $SLURM_TMPDIR/data

# Start training
echo "Executing on lines $1 to $2"
sed -n "$1,$2p" $3 > "tmp$SLURM_JOB_ID.input"
cat "tmp$SLURM_JOB_ID.input" | parallel -j $SLURM_GPUS_PER_NODE --roundrobin 'eval CUDA_VISIBLE_DEVICES=$(({%} - 1)) {} &> results/{#}.out'
rm "tmp$SLURM_JOB_ID.input"

# salloc --nodes=1 --gpus-per-node=2 --cpus-per-task=6 --mem=8000M --time=0-00:30 --account=def-mlecuyer
# sed -n "100,101p" runner.input > "tmp$SLURM_JOB_ID.input"
# cat "tmp$SLURM_JOB_ID.input" | parallel -j $SLURM_GPUS_PER_NODE --roundrobin 'eval CUDA_VISIBLE_DEVICES=$(({%} - 1)) {} &> results/{#}.out'
# rm "tmp$SLURM_JOB_ID.input"