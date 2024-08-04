# create môi trường ảo bằng conda nếu đã có conda sẵn # conda 24.1.2
conda create -n myenv python=3.10
# khởi động môi trường ảo
conda activate myenv
# tắt môi trường ảo
conda deactivate
# cài đặt pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# unzip in window
tar -xf dataset/nq320k.zip

# run genret for nq320k dataset
python run.py --model_name t5-base --code_num 512 --max_length 3 --train_data nq320k/train.json --dev_data nq320k/dev.json --corpus_data nq320k/corpus_lite.json --save_path out/model