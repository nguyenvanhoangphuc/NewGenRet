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

Chú thích: 
- file train.json là một list các item, mỗi item là một list gồm 2 phần tử, phần tử đầu tiên sẽ là question (hoặc query), phần tử thứ 2 là số thứ tự (id) của document mà nó liên quan đến trong bộ corpus_lite.json
+ số lượng item: 307373
- file corpus_lite.json là một list các document, mỗi document là một chuỗi string, thứ tự của document cũng chính là id của document đó.
+ số lượng corpus: 109739
- file train.json.qg.json tương tự như train.json nhưng sẽ có những câu đồng nghĩa của từng câu trong train.json làm số lượng được nhân lên gần 4 lần.
+ số lượng item: 1097390
- file dev.json tương tự như train.json nhưng số lượng ít hơn
+ số lượng item: 7830
