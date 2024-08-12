#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=1

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Run train
python my_run.py --model_name t5-base --code_num 512 --max_length 3 --train_data dataset/ms_marco_bm25_1208/train_retrieval_ms_marco.json --dev_data dataset/ms_marco_bm25_1208/validation_retrieval_ms_marco.json --corpus_data dataset/ms_marco_bm25_1208/legal_corpus_ms_marco.json --save_path out_bm25/model >> output.log