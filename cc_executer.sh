#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=5       # Request GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15      # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=0                 # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-05:00          # DD-HH:MM:SS
#SBATCH --account=def-mlecuyer

module load python/3.7 cuda cudnn

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

# You could also create your environment here, on the local storage ($SLURM_TMPDIR), for better performance. See our docs on virtual environments.

# Prepare data
# mkdir $SLURM_TMPDIR/data
# tar xf ~/projects/def-xxxx/data.tar -C $SLURM_TMPDIR/data

# Start training
cat runner.input | parallel -j5 'CUDA_VISIBLE_DEVICES=$(({%} - 1)) python {} &> {#}.out'
# python main.py $@