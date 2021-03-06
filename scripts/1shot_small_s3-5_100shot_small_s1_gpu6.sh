#!/bin/bash -l

source activate pytorch_p36
#pip install -r ../requirements.txt

export SAVE_PATH=../tf_checkpoints
export DATA_PATH=../data/cogs
export EXAMPLES=1_example
export CUDA_VISIBLE_DEVICES=6
export SAVE_NAME=${EXAMPLES}_small

for SEED in {3..5}
do
	python -u ../train.py -data $DATA_PATH/$EXAMPLES -save_model $SAVE_PATH/${SAVE_NAME}/s$SEED \
	-layers 2 -rnn_size 512 -word_vec_size 512 -transformer_ff 512 -heads 4  \
	-encoder_type transformer -decoder_type transformer -position_encoding \
	-train_steps 30000  -max_generator_batches 2 -dropout 0.1 \
	-batch_size 128 -batch_type sents -normalization sents  -accum_count 4 \
	-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 4000 -learning_rate 2 \
	-max_grad_norm 0 -param_init 0  -param_init_glorot \
	-label_smoothing 0.1 -valid_steps 500 -save_checkpoint_steps 500 -early_stopping 5 \
	-world_size 1 -gpu_ranks 0 -seed $SEED --log_file ../logs/${SAVE_NAME}_s${SEED}.runlog 

	for SPLIT in gen test dev
	do
		python ../translate.py -model $SAVE_PATH/${SAVE_NAME}/s${SEED}_best.pt \
				       -src $DATA_PATH/${SPLIT}_source.txt \
				       -tgt $DATA_PATH/${SPLIT}_target.txt \
				       -output ../preds/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.txt \
				       -replace_unk -verbose -shard_size 0 \
				       -gpu 0 -batch_size 128

		paste $DATA_PATH/${SPLIT}_source.txt $DATA_PATH/${SPLIT}_target.txt ../preds/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.txt > ../preds/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.tsv
	done


done

export EXAMPLES=100_example
export SAVE_NAME=${EXAMPLES}_small
export SEED=1

python -u ../train.py -data $DATA_PATH/$EXAMPLES -save_model $SAVE_PATH/${SAVE_NAME}/s$SEED \
	-layers 2 -rnn_size 512 -word_vec_size 512 -transformer_ff 512 -heads 4  \
	-encoder_type transformer -decoder_type transformer -position_encoding \
	-train_steps 30000  -max_generator_batches 2 -dropout 0.1 \
	-batch_size 128 -batch_type sents -normalization sents  -accum_count 4 \
	-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 4000 -learning_rate 2 \
	-max_grad_norm 0 -param_init 0  -param_init_glorot \
	-label_smoothing 0.1 -valid_steps 500 -save_checkpoint_steps 500 -early_stopping 5 \
	-world_size 1 -gpu_ranks 0 -seed $SEED --log_file ../logs/${SAVE_NAME}_s${SEED}.runlog \

for SPLIT in gen test dev
do
	python ../translate.py -model $SAVE_PATH/${SAVE_NAME}/s${SEED}_best.pt \
                    	       -src $DATA_PATH/${SPLIT}_source.txt \
                    	       -tgt $DATA_PATH/${SPLIT}_target.txt \
                    	       -output ../preds/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.txt \
                    	       -replace_unk -verbose -shard_size 0 \
                    	       -gpu 0 -batch_size 128

	paste $DATA_PATH/${SPLIT}_source.txt $DATA_PATH/${SPLIT}_target.txt ../preds/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.txt > ../preds/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.tsv
done

