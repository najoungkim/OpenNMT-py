#!/bin/bash -l

export DATA_DIR=../data/cogs
source activate pytorch_p36

mkdir $DATA_DIR

python ../preprocess.py -train_src $DATA_DIR/train_source.txt -train_tgt $DATA_DIR/train_target.txt -valid_src $DATA_DIR/dev_source.txt -valid_tgt $DATA_DIR/dev_target.txt -save_data $DATA_DIR/1_example -src_seq_length 5000 -tgt_seq_length 5000 -src_vocab $DATA_DIR/source_vocab.txt -tgt_vocab $DATA_DIR/target_vocab.txt

python ../preprocess.py -train_src $DATA_DIR/train_100_source.txt -train_tgt $DATA_DIR/train_100_target.txt -valid_src $DATA_DIR/dev_source.txt -valid_tgt $DATA_DIR/dev_target.txt -save_data $DATA_DIR/100_example -src_seq_length 5000 -tgt_seq_length 5000 -src_vocab $DATA_DIR/source_vocab.txt -tgt_vocab $DATA_DIR/target_vocab.txt
