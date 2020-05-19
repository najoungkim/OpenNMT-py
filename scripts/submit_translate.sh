#!/bin/bash -l
#SBATCH --job-name=cogs-pred
#SBATCH --time=5:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH -p gpuk80 --gres=gpu:1
#SBATCH -o logs/transformer_translate.log

module load cuda
module load python/3.6-anaconda
module load gcc/5.5.0
module load openmpi/3.1

source activate compgen
export SAVE_PATH=/home-3/nkim43@jhu.edu/work2/nk/exp/emnlp/tf_checkpoints

python translate.py -model $SAVE_PATH/few_s1_step_2500.pt \
                    -src data/cogs_gen_source.txt \
                    -tgt data/cogs_gen_target.txt \
                    -output gen_pred_few_s1.txt \
                    -replace_unk -verbose -shard_size 0 \
                    -gpu 0 -batch_size 256

