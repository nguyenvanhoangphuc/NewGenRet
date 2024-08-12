import json

train_data = "dataset/ms_marco_split/train_retrieval_ms_marco.json"
dev_data = "dataset/ms_marco_split/validation_retrieval_ms_marco.json"
corpus_data = "dataset/ms_marco_split/legal_corpus_ms_marco.json"

corpus = json.load(open(corpus_data))

print("Trước khi xử lý corpus")
# # print(data)
# print(len(corpus[0]["articles"]))
# new_corpus = {}
# for art in corpus[0]["articles"]: 
#     str1 = art['article_id']
#     new_corpus[art['article_id']] = art['title'] + art['text']
# corpus = new_corpus

# print(len(corpus))
# print(corpus[str1])

dict_idx = {}
new_corpus = []
count = 0
for art in corpus[0]["articles"]: 
    # new_corpus[art['article_id']] = art['title'] + art['text']
    new_corpus.append(art['title'] + art['text'])
    dict_idx[art['article_id']] = count
    count += 1
corpus = new_corpus

print("Sau khi xử lý corpus")
print(len(corpus))
print(corpus[0])


data = json.load(open(train_data))

print("Trước khi xử lý train")
# print(data)
print(len(data['items']))
print(data['items'][0])

new_data = []
for item in data['items']: 
    for art in item['relevant_articles']: 
        new_data.append([
            item['question_full'],
            dict_idx[art['law_id'] + art['article_id']]
        ])

data = new_data
print("kết quả sau khi xử lý train")
print(len(data))
print(data[0])


# data = json.load(open(dev_data))

# print("Trước khi xử lý dev")
# # print(data)
# print(len(data['items']))
# print(data['items'][0])

# new_data = []
# for item in data['items']: 
#     for art in item['relevant_articles']: 
#         new_data.append([
#             item['question_full'],
#             art['law_id'] + art['article_id']
#         ])
# data = new_data

# len_dev = len(data)
# seen_split = list(range(len_dev))
# unseen_split = list(range(len_dev))

# print("kết quả sau khi xử lý dev")
# print(len(data))
# print(data[0])
# # print(seen_split)
# print(len(seen_split))
# print(seen_split[0])
# # print(unseen_split)
# print(len(unseen_split))
# print(unseen_split[0])


