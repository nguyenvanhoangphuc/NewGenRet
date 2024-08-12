import json

train_data = "dataset/nq320k_split/train.json"
dev_data = "dataset/nq320k_split/dev.json"
corpus_data = "dataset/nq320k_split/corpus_lite.json"

# data = json.load(open(train_data))

# # print(data)
# print(len(data))
# print(data[0])

# data = json.load(open(dev_data))
# seen_split = json.load(open(f'{dev_data}.seen'))
# unseen_split = json.load(open(f'{dev_data}.unseen'))

# print(len(data))
# print(data[0])
# print(seen_split)
# print(len(seen_split))
# print(seen_split[0])
# print(unseen_split)
# print(len(unseen_split))
# print(unseen_split[0])

corpus = json.load(open(corpus_data))

# print(data)
print(len(corpus))
print(corpus[0])
