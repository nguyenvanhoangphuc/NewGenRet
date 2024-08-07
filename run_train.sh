#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=1

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Chạy train t5-base for english
# python run.py --model_name t5-base --code_num 512 --max_length 3 --train_data nq320k_split/train.json --dev_data nq320k_split/dev.json --corpus_data nq320k_split/corpus_lite.json --save_path out/model
# python generation.py --model_name t5-base --code_num 512 --max_length 3 --train_data nq320k_split/train.json --dev_data nq320k_split/dev.json --corpus_data nq320k_split/corpus_lite.json --save_path out/model

# Chạy train t5-large for japanese
# python run.py --model_name t5-small --code_num 64 --max_length 3 --train_data data_ja/train.json --dev_data data_ja/dev.json --corpus_data data_ja/corpus_lite.json --save_path out_ja/model
python run.py --model_name sonoisa/t5-base-japanese --code_num 64 --max_length 3 --train_data data_ja/train.json --dev_data data_ja/dev.json --corpus_data data_ja/corpus_lite.json --save_path out_ja/model
# nohup python run.py --model_name sonoisa/t5-base-japanese --code_num 64 --max_length 3 --train_data data_ja/train.json --dev_data data_ja/dev.json --corpus_data data_ja/corpus_lite.json --save_path out_ja/model  > output.log 2>&1 &