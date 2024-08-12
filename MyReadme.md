# create môi trường ảo bằng conda nếu đã có conda sẵn # conda 24.1.2
conda create -n myenv python=3.10
# khởi động môi trường ảo
conda activate myenv
# tắt môi trường ảo
conda deactivate
# cài đặt pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# unzip in window (không cần thiết nếu không train toàn bộ nq320k)
tar -xf dataset/nq320k.zip

# run train for nq320k split using bm25 dataset (copy dòng lệnh trên bỏ qua vị trí run train trong run_train.sh)
# ban đầu
python run.py --model_name t5-base --code_num 512 --max_length 3 --train_data dataset/ms_marco_bm25_1208/train_retrieval_ms_marco.json --dev_data dataset/ms_marco_bm25_1208/validation_retrieval_ms_marco.json --corpus_data dataset/ms_marco_bm25_1208/legal_corpus_ms_marco.json --save_path out_bm25/model >> output.log
# Phuc tách thành các file chuyên biệt
python my_run.py --model_name t5-base --code_num 512 --max_length 3 --train_data dataset/ms_marco_bm25_1208/train_retrieval_ms_marco.json --dev_data dataset/ms_marco_bm25_1208/validation_retrieval_ms_marco.json --corpus_data dataset/ms_marco_bm25_1208/legal_corpus_ms_marco.json --save_path out_bm25/model >> output.log

# Về format của dataset nq320k: (code repo này đã thay đổi format đầu vào nên chạy tập này sẽ lỗi)
- file train.json là một list các item, mỗi item là một list gồm 2 phần tử, phần tử đầu tiên sẽ là question (hoặc query), phần tử thứ 2 là số thứ tự (id) của document mà nó liên quan đến trong bộ corpus_lite.json
+ số lượng item: 307373
- file corpus_lite.json là một list các document, mỗi document là một chuỗi string, thứ tự của document cũng chính là id của document đó.
+ số lượng corpus: 109739
- file train.json.qg.json tương tự như train.json nhưng sẽ có những câu đồng nghĩa của từng câu trong train.json làm số lượng được nhân lên gần 4 lần.
+ số lượng item: 1097390
- file dev.json tương tự như train.json nhưng số lượng ít hơn
+ số lượng item: 7830

# Format dataset chạy được 
- phải thuộc dạng dataset legal corpus gồm 4 file train, valid, test, legal_corpus.

# Tinh chỉnh epochs huấn luyện
lần lượt tạo ra các thư mục model-1-pre, model-1, model-2-pre, model-2, model-3-pre, model-3, model-3-fit
số epochs huấn luyện có thể được quy định, mặc định thì model-1-pre là 1 epoch, model-2-pre, model-3-pre là 10 epochs, 
model-1, model-2, model-3 mặc định là 200 mà vì mình lưu step là 9 nên phải là bội của 9 + 1, hiện tại trong code là 199 (dòng 658 file my_run.py)
model-3-fit mặc định là 1000 thoã mãn bội của 9 + 1 mà nó hơi nhiều nên giảm xuống hiện tại trong code là 541 (dòng 671 file my_run.py)


# Chỉnh batch_size:
- Tìm kiếm in_batch_size dòng 635, chỉnh từ 32 lên 128 nếu như có GPU dư