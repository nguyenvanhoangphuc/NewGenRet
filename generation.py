import copy
from abc import ABC

from transformers import T5EncoderModel, AutoTokenizer, AutoModelForSeq2SeqLM
from torch.nn.utils.rnn import pad_sequence
from transformers import Adafactor, get_linear_schedule_with_warmup, get_constant_schedule
from torch.optim import AdamW
from accelerate import Accelerator
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from transformers.trainer_pt_utils import get_parameter_names
# old
# from transformers.generation_utils import GenerationMixin
# new
from transformers.generation import GenerationMixin
from torch import nn, Tensor
import torch.distributed as dist
from typing import Optional, Union, List, Dict, Any, Tuple
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers.modeling_outputs import ModelOutput
import torch.nn.functional as F
from utils.io import read_pkl, write_pkl, read_file
from collections import defaultdict
from copy import deepcopy
import numpy as np
import json
import faiss
import torch
import os
import argparse
import time
from tqdm import tqdm
import torch

class Tree:
    def __init__(self):
        self.root = dict()

    def set(self, path):
        pointer = self.root
        for i in path:
            if i not in pointer:
                pointer[i] = dict()
            pointer = pointer[i]

    def set_all(self, path_list):
        for path in tqdm(path_list):
            self.set(path)

    def find(self, path):
        if isinstance(path, torch.Tensor):
            path = path.cpu().tolist()
        pointer = self.root
        for i in path:
            if i not in pointer:
                return []
            pointer = pointer[i]
        return list(pointer.keys())

    def __call__(self, batch_id, path):
        return self.find(path)

def safe_load(model, file):
    state_dict = torch.load(file, map_location=lambda storage, loc: storage)
    model_state_dict_keys = list(model.state_dict().keys())
    new_state_dict_keys = list(state_dict.keys())
    new_keys_in_new = [k for k in new_state_dict_keys if k not in model_state_dict_keys]
    no_match_keys_of_model = [k for k in model_state_dict_keys if k not in new_state_dict_keys]
    # size_not_match = [k for k,v in state_dict.items() if model_state_dict_keys[k]]
    print('##', model._get_name(), '# new keys in file:', new_keys_in_new, '# no match keys:', no_match_keys_of_model)
    model.load_state_dict(state_dict, strict=False)

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


@dataclass
class QuantizeOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    all_dense_embed: Optional[torch.FloatTensor] = None
    continuous_embeds: Optional[torch.FloatTensor] = None
    quantized_embeds: Optional[torch.FloatTensor] = None
    discrete_codes: Optional[torch.LongTensor] = None
    probability: Optional[torch.FloatTensor] = None
    code_logits: Optional[torch.FloatTensor] = None


@torch.no_grad()
def sinkhorn_raw(out: Tensor, epsilon: float,
                 sinkhorn_iterations: int,
                 use_distrib_train: bool):
    Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from our paper

    B = Q.shape[1]
    K = Q.shape[0]  # how many prototypes
    # make the matrix sums to 1
    sum_Q = torch.clamp(torch.sum(Q), min=1e-5)
    if use_distrib_train:
        B *= dist.get_world_size()
        dist.all_reduce(sum_Q)
    Q /= sum_Q
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.clamp(torch.sum(Q, dim=1, keepdim=True), min=1e-5)
        if use_distrib_train:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.clamp(torch.sum(torch.sum(Q, dim=0, keepdim=True), dim=1, keepdim=True), min=1e-5)
        Q /= B
    Q *= B
    return Q.t()

class Model(nn.Module, GenerationMixin, ABC):
    def __init__(self, model, use_constraint: bool, sk_epsilon: float = 0.03, sk_iters: int = 100, code_length=1,
                 zero_inp=False, code_number=10):
        super().__init__()
        self.model = model
        self.config = model.config
        self.generation_config = model.generation_config
        # self.base_model_prefix = 't5'
        # self._supports_cache_class = True
        self.main_input_name = model.main_input_name
        self.get_encoder = model.get_encoder
        self.device = model.device
        self.prepare_inputs_for_generation = model.prepare_inputs_for_generation
        self.can_generate = lambda: True
        hidden_size = model.config.hidden_size

        self.use_constraint, self.sk_epsilon, self.sk_iters = use_constraint, sk_epsilon, sk_iters

        # Codebook of each time step
        self.centroids = nn.ModuleList([nn.Linear(hidden_size, code_number, bias=False) for _ in range(code_length)])
        self.centroids.requires_grad_(True)

        # Code embedding (input to the decoder)
        self.code_embedding = nn.ModuleList([nn.Embedding(code_number, hidden_size) for _ in range(code_length)])
        self.code_embedding.requires_grad_(True)

        self.code_length = code_length
        self.zero_inp = zero_inp
        self.code_number = code_number

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_outputs=None, **kwargs):
        return {"decoder_input_ids": input_ids, "encoder_outputs": encoder_outputs, "attention_mask": attention_mask}

    @torch.no_grad()
    def quantize(self, probability, use_constraint=None):
        # batchsize_per_device = len(continuous_embeds)
        # distances = ((continuous_embeds.reshape(batchsize_per_device, self.config.MCQ_M, 1, -1).transpose(0,1) -
        #               self.centroids.unsqueeze(1)) ** 2).sum(-1)  # M, bs', K
        distances = -probability
        use_constraint = self.use_constraint if use_constraint is None else use_constraint
        # raw_code = torch.argmin(distances, dim=-1)
        # print('In', torch.argmin(distances, dim=-1))
        if not use_constraint:
            codes = torch.argmin(distances, dim=-1)  # M, bs
        else:
            distances = self.center_distance_for_constraint(distances)  # to stablize
            # avoid nan
            distances = distances.double()
            # Q = sinkhorn_algorithm(
            #     -distances.transpose(1, 2),
            #     self.sk_epsilon,
            #     self.sk_iters,
            #     use_distrib_train=dist.is_initialized()
            # ).transpose(1, 2)  # M-B-K
            Q = sinkhorn_raw(
                -distances,
                self.sk_epsilon,
                self.sk_iters,
                use_distrib_train=dist.is_initialized()
            )  # B-K
            codes = torch.argmax(Q, dim=-1)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
        # print('Out', codes)
        # print('Equal', (raw_code == codes).float().mean())
        # codes = codes.t()  # bs, M
        # input('>')
        return codes

    def decode(self, codes, centroids=None):
        M = codes.shape[1]
        if centroids is None:
            centroids = self.centroids
        if isinstance(codes, torch.Tensor):
            assert isinstance(centroids, torch.Tensor)
            first_indices = torch.arange(M).to(codes.device)
            first_indices = first_indices.expand(*codes.shape).reshape(-1)
            quant_embeds = centroids[first_indices, codes.reshape(-1)].reshape(len(codes), -1)
        elif isinstance(codes, np.ndarray):
            if isinstance(centroids, torch.Tensor):
                centroids = centroids.detach().cpu().numpy()
            first_indices = np.arange(M)
            first_indices = np.tile(first_indices, len(codes))
            quant_embeds = centroids[first_indices, codes.reshape(-1)].reshape(len(codes), -1)
        else:
            raise NotImplementedError()
        return quant_embeds

    def embed_decode(self, codes, centroids=None):
        if centroids is None:
            centroids = self.centroids[-1]
        quant_embeds = F.embedding(codes, centroids.weight)
        return quant_embeds

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: M, bs, K
        max_distance = distances.max()
        min_distance = distances.min()
        if dist.is_initialized():
            dist.all_reduce(max_distance, torch.distributed.ReduceOp.MAX)
            dist.all_reduce(min_distance, torch.distributed.ReduceOp.MIN)
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert torch.all(amplitude > 0)
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, aux_ids=None, return_code=False,
                return_quantized_embedding=False, use_constraint=None, encoder_outputs=None, **kwargs):
        if decoder_input_ids is None or self.zero_inp:
            decoder_input_ids = torch.zeros(input_ids.size(0), self.code_length).long().to(input_ids.device)

        # decoder_inputs_embeds = self.code_embedding(decoder_input_ids)

        decoder_inputs_embeds = []
        for i in range(min(decoder_input_ids.size(1), len(self.code_embedding))):
            code_embedding = self.code_embedding[i]
            decoder_inputs_embeds.append(code_embedding(decoder_input_ids[:, i]))
        decoder_inputs_embeds = torch.stack(decoder_inputs_embeds, dim=1)

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=True,
            encoder_outputs=encoder_outputs
        )
        decoder_outputs = model_outputs.decoder_hidden_states[-1]
        all_dense_embed = decoder_outputs.view(decoder_outputs.size(0), -1).contiguous()
        dense_embed = decoder_outputs[:, -1].contiguous()

        code_logits = []
        for i in range(min(decoder_input_ids.size(1), len(self.code_embedding))):
            centroid = self.centroids[i]
            code_logits.append(centroid(decoder_outputs[:, i]))
        code_logits = torch.stack(code_logits, dim=1)
        # code_logits = self.centroids(decoder_outputs)

        probability = code_logits[:, -1].contiguous()
        # probability = torch.mm(dense_embed, self.centroids.transpose(0, 1))
        discrete_codes = self.quantize(probability, use_constraint=use_constraint)

        if aux_ids is None:
            aux_ids = discrete_codes

        quantized_embeds = self.embed_decode(aux_ids) if return_quantized_embedding else None

        if self.code_length == 1:
            return_code_logits = None
        else:
            return_code_logits = code_logits[:, :-1].contiguous()

        quant_output = QuantizeOutput(
            logits=code_logits,
            all_dense_embed=all_dense_embed,
            continuous_embeds=dense_embed,
            quantized_embeds=quantized_embeds,
            discrete_codes=discrete_codes,
            probability=probability,
            code_logits=return_code_logits,
        )
        return quant_output

def test(config):
    model_name = config.get('model_name', 't5-base')
    code_num = config.get('code_num', 512)
    code_length = config.get('code_length', 1)
    prev_id = config.get('prev_id', None)
    save_path = config.get('save_path', None)
    batch_size = 32
    epochs = config.get('epochs', 100)

    dev_data = config.get('dev_data', config.get('dev_data'))
    corpus_data = config.get('corpus_data', config.get('corpus_data'))

    data = json.load(open(dev_data))
    corpus = json.load(open(corpus_data))

    t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = Model(model=t5, use_constraint=False, code_length=code_length, zero_inp=False, code_number=code_num)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ids = None
    if prev_id is not None:
        corpus_ids = [[0, *line] for line in json.load(open(prev_id))]
    else:
        corpus_ids = [[0]] * len(corpus)
    aux_ids = None

    dataset = BiDataset(data=data, corpus=corpus, tokenizer=tokenizer, max_doc_len=128, max_q_len=32, ids=corpus_ids,
                        aux_ids=aux_ids)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
                                              shuffle=False, num_workers=16)
    model = model.cuda()
    model.eval()

    seen_split = json.load(open(f'{dev_data}.seen'))
    print("len(seen_split)", len(seen_split))
    unseen_split = json.load(open(f'{dev_data}.unseen'))
    print("len(unseen_split)", len(unseen_split))

    for epoch in range(epochs):
        if not os.path.exists(f'{save_path}/{epoch}.pt'):
            continue
        print(f'Test {save_path}/{epoch}.pt')

        corpus_ids = [[0, *line] for line in json.load(open(f'{save_path}/{epoch}.pt.code'))]
        safe_load(model, f'{save_path}/{epoch}.pt')
        tree = Tree()
        tree.set_all(corpus_ids)

        tk0 = tqdm(data_loader, total=len(data_loader))
        acc = []
        output_all = []
        with torch.no_grad():
            for batch in tk0:
                batch = {k: v.cuda() for k, v in batch.items() if v is not None}
                top_k = 10
                output = model.model.generate(
                    input_ids=batch['query'].cuda(),
                    attention_mask=batch['query'].ne(0).cuda(),
                    max_length=code_length + 1,
                    num_beams=top_k,
                    num_return_sequences=top_k,
                    prefix_allowed_tokens_fn=tree
                )
                beam = []
                new_output = []
                for line in output:
                    if len(beam) >= top_k:
                        new_output.append(beam)
                        beam = []
                    beam.append(line.cpu().tolist())
                new_output.append(beam)
                output_all.extend(new_output)

        query_ids = [x[1] for x in data]

        docid_to_doc = defaultdict(list)
        for i, item in enumerate(corpus_ids):
            docid_to_doc[str(item)].append(i)
        predictions = []
        for line in output_all:
            new_line = []
            for s in line:
                s = str(s)
                if s not in docid_to_doc:
                    continue
                tmp = docid_to_doc[s]
                # np.random.shuffle(tmp)
                new_line.extend(tmp)
                if len(new_line) > 100:
                    break
            predictions.append(new_line)

        from eval import eval_all
        print('Test', eval_all(predictions, query_ids))
        print("len(predictions)", len(predictions))
        print(eval_all([predictions[j] for j in seen_split], [query_ids[j] for j in seen_split]))
        print(eval_all([predictions[j] for j in unseen_split], [query_ids[j] for j in unseen_split]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/data03/sunweiwei-slurm/huggingface/t5-base')
    parser.add_argument('--code_num', type=int, default=512)
    parser.add_argument('--max_length', type=int, default=3)
    parser.add_argument('--train_data', type=str, default='dataset/nq320k/train.json')
    parser.add_argument('--dev_data', type=str, default='dataset/nq320k/dev.json')
    parser.add_argument('--corpus_data', type=str, default='dataset/nq320k/corpus_lite.json')
    parser.add_argument('--save_path', type=str, default='out/model')
    args = parser.parse_args()

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    return args


def main():
    # đọc các tham số từ command line
    args = parse_args()
    # tạo config là một bản sao của args
    config = copy.deepcopy(vars(args))
    # khởi tạo checkpoint ban đầu là None
    checkpoint = None
    # lặp qua các giá trị từ 0 đến max_length-1 (3-1)
    for loop in range(args.max_length):
        # cập nhật save_path, code_length theo loop
        config['save_path'] = args.save_path + f'-{loop + 1}-pre'
        config['code_length'] = loop + 1
        # prev_model là checkpoint trước đó, đi kèm prev_id là file code của checkpoint trước đó (file này sẽ được tạo ra sau khi chạy train)
        config['prev_model'] = checkpoint
        config['prev_id'] = f'{checkpoint}.code' if checkpoint is not None else None
        # lần đầu tiên chỉ chạy 1 epoch, sau đó chạy 10 epoch
        config['epochs'] = 1 if loop == 0 else 10
        config['loss_w'] = 1
        # chạy train
        # checkpoint = train(config)
        checkpoint = 'out/model-1-pre/0.pt'
        # test_dr(config)

        config['save_path'] = args.save_path + f'-{loop + 1}'
        config['prev_model'] = checkpoint
        config['codebook_init'] = f'{checkpoint}.kmeans.{args.code_num}'
        config['epochs'] = 10   # default: 200
        config['loss_w'] = 2
        # checkpoint = train(config)
        checkpoint = 'out/model-1/9.pt'
        # test_dr(config)

        test(config)

    # loop = args.max_length
    # config['save_path'] = args.save_path + f'-{loop}-fit'
    # config['code_length'] = loop + 1
    # config['prev_model'] = checkpoint
    # add_last(f'{checkpoint}.code', args.code_num, f'{checkpoint}.code.last')
    # config['prev_id'] = f'{checkpoint}.code.last'
    # config['epochs'] = 10     # default: 1000
    # config['loss_w'] = 3
    # checkpoint = train(config)
    # test_dr(config)
    # test(config)
    # print(checkpoint)


if __name__ == '__main__':
    main()
