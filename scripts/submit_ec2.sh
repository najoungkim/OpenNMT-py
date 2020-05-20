#!/bin/bash -l

source activate pytorch_p36
#pip install -r ../requirements.txt

export SAVE_PATH=../tf_checkpoints
export DATA_PATH=../data/cogs
export EXAMPLES=1_example
export SEED=2
export SAVE_NAME=${EXAMPLES}

#python -u ../train.py -data $DATA_PATH/$EXAMPLES -save_model $SAVE_PATH/${SAVE_NAME}/s$SEED \
#        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
#        -encoder_type transformer -decoder_type transformer -position_encoding \
#        -train_steps 30000  -max_generator_batches 2 -dropout 0.1 \
#        -batch_size 128 -batch_type sents -normalization sents  -accum_count 4 \
#        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 4000 -learning_rate 2 \
#        -max_grad_norm 0 -param_init 0  -param_init_glorot \
#        -label_smoothing 0.1 -valid_steps 500 -save_checkpoint_steps 500 -early_stopping 5 \
#        -world_size 1 -gpu_ranks 0 -seed $SEED --log_file ../logs/${SAVE_NAME}_s${SEED}.runlog \

#mkdir ../preds

#python ../translate.py -model $SAVE_PATH/${SAVE_NAME}/s${SEED}_step_3000.pt \
#                    -src $DATA_PATH/gen_source.txt \
#                    -tgt $DATA_PATH/gen_target.txt \
#                    -output ../preds/gen_pred_${SAVE_NAME}_s${SEED}.txt \
#                    -replace_unk -verbose -shard_size 0 \
#                    -gpu 0 -batch_size 128

#paste $DATA_PATH/gen_source.txt $DATA_PATH/gen_target.txt ../preds/gen_pred_${SAVE_NAME}_s${SEED}.txt > ../preds/gen_pred_${SAVE_NAME}_s${SEED}.tsv


python ../translate.py -model $SAVE_PATH/${SAVE_NAME}/s${SEED}_step_3000.pt \
                    -src $DATA_PATH/test_source.txt \
                    -tgt $DATA_PATH/test_target.txt \
                    -output ../preds/test_pred_${SAVE_NAME}_s${SEED}.txt \
                    -replace_unk -verbose -shard_size 0 \
                    -gpu 0 -batch_size 128

paste $DATA_PATH/test_source.txt $DATA_PATH/test_target.txt ../preds/test_pred_${SAVE_NAME}_s${SEED}.txt > ../preds/test_pred_${SAVE_NAME}_s${SEED}.tsv
