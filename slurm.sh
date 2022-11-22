#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 4
#SBATCH --time 2-00:00:00
#SBATCH --output ./logs/%j.out
#

singularity exec --nv --bind $PWD:/workspace \
  --bind $(readlink -f data/nuimages):$(readlink -f data/nuimages) \
  --bind $(readlink -f data/nuscenes):$(readlink -f data/nuscenes) \
  --pwd /workspace \
  --env PYTHONPATH=/workspace/:/workspace/ConditionalDETR/ \
  future-od.sif \
  python3 -u -m torch.distributed.launch --nproc_per_node=4 --master_port=$RANDOM \
  $@ --distributed

#
#EOF
