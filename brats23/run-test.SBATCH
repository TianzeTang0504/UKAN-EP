#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200GB
#SBATCH	--gres=gpu:rtx8000:1
#SBATCH --time=48:00:00
#SBATCH --output=ukan_%j.txt

module purge

singularity exec --nv \
            --overlay /scratch/tt2631/pytorch_1/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python main.py"

