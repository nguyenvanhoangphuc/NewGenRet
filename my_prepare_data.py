
from abc import ABC
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from typing import Optional
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch


@dataclass
class QuantizeOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    all_dense_embed: Optional[torch.FloatTensor] = None
    continuous_embeds: Optional[torch.FloatTensor] = None
    quantized_embeds: Optional[torch.FloatTensor] = None
    discrete_codes: Optional[torch.LongTensor] = None
    probability: Optional[torch.FloatTensor] = None
    code_logits: Optional[torch.FloatTensor] = None


class BiDataset(Dataset, ABC):
    def __init__(self, data, corpus, tokenizer, max_doc_len=512, max_q_len=128, ids=None, batch_size=1, aux_ids=None):
        self.data = data
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_doc_len = max_doc_len
        self.max_q_len = max_q_len
        self.ids = ids
        self.batch_size = batch_size

        if self.batch_size != 1:
            ids_to_item = defaultdict(list)
            for i, item in enumerate(self.data):
                ids_to_item[str(ids[item[1]])].append(i)
            for key in ids_to_item:
                np.random.shuffle(ids_to_item[key])
            self.ids_to_item = ids_to_item
        else:
            self.ids_to_item = None
        self.aux_ids = aux_ids

    def getitem(self, item):
        queries, doc_id = self.data[item]
        if isinstance(queries, list):
            query = np.random.choice(queries)
        else:
            query = queries

        while isinstance(doc_id, list):
            doc_id = doc_id[0]

        doc = self.corpus[doc_id]
        if self.ids is None:
            ids = [0]
        else:
            ids = self.ids[doc_id]
        if self.aux_ids is None:
            aux_ids = -100
        else:
            aux_ids = self.aux_ids[doc_id]
        return (torch.tensor(self.tokenizer.encode(query, truncation=True, max_length=self.max_q_len)),
                torch.tensor(self.tokenizer.encode(doc, truncation=True, max_length=self.max_doc_len)),
                ids, aux_ids)

    def __getitem__(self, item):
        if self.batch_size == 1:
            return [self.getitem(item)]
        else:
            # item_set = self.ids_to_item[str(self.ids[self.data[item][1]])]
            # new_item_set = [item] + [i for i in item_set if i != item]
            # work_item_set = new_item_set[:self.batch_size]
            # new_item_set = new_item_set[self.batch_size:] + work_item_set
            # self.ids_to_item[str(self.ids[self.data[item][1]])] = new_item_set

            item_set = deepcopy(self.ids_to_item[str(self.ids[self.data[item][1]])])
            np.random.shuffle(item_set)
            item_set = [item] + [i for i in item_set if i != item]
            work_item_set = item_set[:self.batch_size]

            if len(work_item_set) < self.batch_size:
                rand_item_set = np.random.randint(len(self), size=self.batch_size * 2)
                rand_item_set = [i for i in rand_item_set if i != item]
                work_item_set = work_item_set + rand_item_set
                work_item_set = work_item_set[:self.batch_size]

            collect = []
            for item in work_item_set:
                query, doc, ids, aux_ids = self.getitem(item)
                collect.append((query, doc, ids, aux_ids))
            return collect

    def __len__(self):
        return len(self.data)

    def collate_fn(self, data):
        data = sum(data, [])
        query, doc, ids, aux_ids = zip(*data)
        query = pad_sequence(query, batch_first=True, padding_value=0)
        doc = pad_sequence(doc, batch_first=True, padding_value=0)
        ids = torch.tensor(ids)
        if self.aux_ids is None:
            aux_ids = None
        else:
            aux_ids = torch.tensor(aux_ids)
        return {
            'query': query,
            'doc': doc,
            'ids': ids,
            'aux_ids': aux_ids
        }