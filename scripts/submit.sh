#!/bin/bash -l
#SBATCH --job-name=transformer
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH -p gpuk80 --gres=gpu:1
#SBATCH -o logs/transformer_verb_prims_s1.log

module load cuda
module load python/3.6-anaconda
module load gcc/5.5.0
module load openmpi/3.1

source activate compgen
export SAVE_PATH=/home-3/nkim43@jhu.edu/work2/nk/exp/emnlp/tf_checkpoints
export SEED=1

python -u train.py -data data/cogs_verb_prims -save_model $SAVE_PATH/verb_prims_s$SEED \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 30000  -max_generator_batches 2 -dropout 0.1 \
        -batch_size 128 -batch_type sents -normalization sents  -accum_count 4 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 4000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 500 -save_checkpoint_steps 500 -early_stopping 5 \
        -world_size 1 -gpu_ranks 0 -seed $SEED
