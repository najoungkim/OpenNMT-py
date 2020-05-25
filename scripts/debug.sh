#!/bin/bash -l

source activate pytorch_p36
#pip install -r ../requirements.txt

export SAVE_PATH=../tf_checkpoints
export DATA_PATH=../data/cogs
export EXAMPLES=100_example
export CUDA_VISIBLE_DEVICES=0
export SAVE_NAME=${EXAMPLES}_small

for SEED in {2..2}
do

	for SPLIT in gen
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


