import copy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import get_constant_schedule
from torch.optim import AdamW
from accelerate import Accelerator
from transformers.trainer_pt_utils import get_parameter_names
from torch import nn, Tensor
import torch.distributed as dist
from tqdm import tqdm
import torch.nn.functional as F
from utils.io import read_pkl, write_pkl
from collections import defaultdict
import numpy as np
import json
import faiss
import torch
import os
import argparse
import time
from my_tree import Tree
from my_prepare_data import QuantizeOutput, BiDataset
from my_model import Model
from my_save_load import safe_load, safe_load_embedding, safe_save
from my_metrics import conflict, balance
from my_norm import norm_by_prefix
from my_kmean import constrained_km


def get_optimizer(model, lr):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters and "centroids" not in n],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       n not in decay_parameters and "centroids" not in n],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if "centroids" in n],
            "weight_decay": 0.0,
            'lr': lr * 20
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr)
    return optimizer


class OurTrainer:
    @staticmethod
    def _gather_tensor(t: Tensor, local_rank):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[local_rank] = t
        return all_tensors

    @staticmethod
    def gather_tensors(t: Tensor, local_rank=None):
        if local_rank is None:
            local_rank = dist.get_rank()
        return torch.cat(OurTrainer._gather_tensor(t, local_rank))

    @staticmethod
    @torch.no_grad()
    def test_step(model: Model, batch, use_constraint=None):
        query_outputs: QuantizeOutput = model(input_ids=batch['query'], attention_mask=batch['query'].ne(0),
                                              decoder_input_ids=batch['ids'],
                                              aux_ids=None, return_code=False,
                                              return_quantized_embedding=False, use_constraint=use_constraint)
        doc_outputs: QuantizeOutput = model(input_ids=batch['doc'], attention_mask=batch['doc'].ne(0),
                                            decoder_input_ids=batch['ids'],
                                            aux_ids=None, return_code=False,
                                            return_quantized_embedding=False, use_constraint=use_constraint)
        return query_outputs, doc_outputs

    @staticmethod
    def simple_train_step(model: Model, batch, gathered=None):
        query_outputs: QuantizeOutput = model(input_ids=batch['query'], attention_mask=batch['query'].ne(0),
                                              decoder_input_ids=batch['ids'])
        # doc_outputs: QuantizeOutput = model(input_ids=batch['doc'], attention_mask=batch['doc'].ne(0),
        #                                     decoder_input_ids=batch['ids'])

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            code_number = model.module.code_number
        else:
            code_number = model.code_number
        # code_number = 10
        query_code_loss = F.cross_entropy(query_outputs.code_logits.view(-1, code_number),
                                          batch['ids'][:, 1:].reshape(-1))
        # doc_code_loss = F.cross_entropy(doc_outputs.code_logits.view(-1, code_number),
        #                                 batch['ids'][:, 1:].reshape(-1))
        query_prob = query_outputs.probability
        aux_query_code_loss = F.cross_entropy(query_prob, batch['aux_ids'])
        code_loss = query_code_loss
        return dict(
            loss=query_code_loss + aux_query_code_loss,
        )

    @staticmethod
    def train_step(model: Model, batch, gathered=None):
        query_outputs: QuantizeOutput = model(input_ids=batch['query'], attention_mask=batch['query'].ne(0),
                                              decoder_input_ids=batch['ids'],
                                              aux_ids=batch['aux_ids'], return_code=True,
                                              return_quantized_embedding=True)
        doc_outputs: QuantizeOutput = model(input_ids=batch['doc'], attention_mask=batch['doc'].ne(0),
                                            decoder_input_ids=batch['ids'],
                                            aux_ids=batch['aux_ids'], return_code=True,
                                            return_quantized_embedding=True)
        query_embeds = query_outputs.continuous_embeds
        doc_embeds = doc_outputs.continuous_embeds
        codes_doc = doc_outputs.discrete_codes
        quant_doc_embeds = doc_outputs.quantized_embeds
        query_prob = query_outputs.probability
        doc_prob = doc_outputs.probability

        query_all_embeds = query_outputs.all_dense_embed
        doc_all_embeds = doc_outputs.all_dense_embed

        if gathered is None:
            gathered = dist.is_initialized()

        cl_loss = OurTrainer.compute_contrastive_loss(query_embeds, doc_embeds, gathered=False)  # retrieval

        all_cl_loss = OurTrainer.compute_contrastive_loss(query_all_embeds, doc_all_embeds,
                                                          gathered=dist.is_initialized())  # retrieval (used when dist)

        # cl_d_loss = OurTrainer.compute_contrastive_loss(doc_embeds, query_embeds, gathered=gathered)
        # cl_loss = cl_q_loss + cl_d_loss

        # mse_loss = 0
        cl_dd_loss = OurTrainer.compute_contrastive_loss(
            quant_doc_embeds + doc_embeds - doc_embeds.detach(), doc_embeds.detach(), gathered=False)  # reconstruction
        # mse_loss = ((quant_doc_embeds - doc_embeds.detach()) ** 2).sum(-1).mean()

        # codes_doc_cpu = codes_doc.cpu().tolist()
        # print(balance(codes_doc_cpu))
        # print(codes_doc)
        query_ce_loss = F.cross_entropy(query_prob, codes_doc.detach())  # commitment
        doc_ce_loss = F.cross_entropy(doc_prob, codes_doc.detach())  # commitment
        ce_loss = query_ce_loss + doc_ce_loss  # commitment

        code_loss = 0
        if query_outputs.code_logits is not None:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                code_number = model.module.code_number
            else:
                code_number = model.code_number
            query_code_loss = F.cross_entropy(query_outputs.code_logits.view(-1, code_number),
                                              batch['ids'][:, 1:].reshape(-1))
            doc_code_loss = F.cross_entropy(doc_outputs.code_logits.view(-1, code_number),
                                            batch['ids'][:, 1:].reshape(-1))
            code_loss = query_code_loss + doc_code_loss  # commitment
        if batch['aux_ids'] is not None:
            aux_query_code_loss = F.cross_entropy(query_prob, batch['aux_ids'])
            aux_doc_code_loss = F.cross_entropy(doc_prob, batch['aux_ids'])
            aux_code_loss = aux_query_code_loss + aux_doc_code_loss  # commitment on last token
            # print('Q', aux_query_code_loss.item(), 'D', aux_doc_code_loss.item())
            if aux_code_loss.isnan():
                aux_code_loss = 0
        else:
            aux_code_loss = 0

        if dist.is_initialized():
            all_doc_embeds = OurTrainer.gather_tensors(doc_embeds)
            global_mean_doc_embeds = all_doc_embeds.mean(dim=0)
            local_mean_doc_embeds = doc_embeds.mean(dim=0)
            clb_loss = F.mse_loss(local_mean_doc_embeds, global_mean_doc_embeds.detach())  # not used
        else:
            clb_loss = 0

        return dict(
            cl_loss=cl_loss,
            all_cl_loss=all_cl_loss,
            mse_loss=0,
            ce_loss=ce_loss,
            code_loss=code_loss,
            aux_code_loss=aux_code_loss,
            cl_dd_loss=cl_dd_loss,
            clb_loss=clb_loss
        )

    @staticmethod
    def compute_contrastive_loss(query_embeds, doc_embeds, gathered=True):
        if gathered:
            gathered_query_embeds = OurTrainer.gather_tensors(query_embeds)
            gathered_doc_embeds = OurTrainer.gather_tensors(doc_embeds)
        else:
            gathered_query_embeds = query_embeds
            gathered_doc_embeds = doc_embeds
        effective_bsz = gathered_query_embeds.size(0)
        labels = torch.arange(effective_bsz, dtype=torch.long, device=query_embeds.device)
        similarities = torch.matmul(gathered_query_embeds, gathered_doc_embeds.transpose(0, 1))
        # similarities = similarities
        co_loss = F.cross_entropy(similarities, labels)
        return co_loss



def train(config):
    accelerator = Accelerator(gradient_accumulation_steps=1)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    model_name = config.get('model_name', 't5-base')
    code_num = config.get('code_num', 512)
    code_length = config.get('code_length', 1)
    prev_model = config.get('prev_model', None)
    prev_id = config.get('prev_id', None)
    save_path = config.get('save_path', None)

    train_data = config.get('train_data', 'dataset/nq320k/train.json')
    corpus_data = config.get('corpus_data', 'dataset/nq320k/corpus_lite.json')
    epochs = config.get('epochs', 100)
    in_batch_size = config.get('batch_size', 128)    # default: 128
    end_epoch = epochs

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    save_step = 9
    batch_size = 1
    lr = 5e-4
    # load model t5-base để train
    accelerator.print(save_path)
    t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = Model(model=t5, code_length=code_length,
                  use_constraint=True, sk_epsilon=1, zero_inp=False, code_number=code_num)
    # lấy các thông số của model trước đó
    if prev_model is not None:
        safe_load(model.model, f'{prev_model}.model')
        safe_load(model.centroids, f'{prev_model}.centroids')
        safe_load_embedding(model.code_embedding, f'{prev_model}.embedding')

    if config.get('codebook_init', None) is not None:
        model.centroids[-1].weight.data = torch.tensor(read_pkl(config.get('codebook_init')))

    for i in range(code_length - 1):
        model.centroids[i].requires_grad_(False)
    # load tokenizer của t5-base
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # load train_data, corpus_data
    data = json.load(open(train_data))
    # # load train.json.qg.json là file chứa các câu hỏi nhưng nhiều câu hỏi cho một doc
    # data.extend(json.load(open(f'{train_data}.qg.json')))
    # load corpus_lite.json là file chứa các document mà được file train liên kết đến
    corpus = json.load(open(corpus_data))
    # xử lý data để đưa vào train
    dict_idx = {}
    new_corpus = []
    count = 0
    for art in corpus[0]["articles"]: 
        # new_corpus[art['article_id']] = art['title'] + art['text']
        new_corpus.append(art['title'] + art['text'])
        dict_idx[art['article_id']] = count
        count += 1
    corpus = new_corpus

    new_data = []
    for item in data['items']: 
        for art in item['relevant_articles']: 
            new_data.append([
                item['question_full'],
                dict_idx[art['law_id'] + art['article_id']]
            ])
    data = new_data
    # tạo grouped_data gồm các key, value là doc_id và các query cùng liên quan đến doc_id đó
    grouped_data = defaultdict(list)
    for i, item in enumerate(data):
        query, docid = item
        if isinstance(docid, list):
            docid = docid[0]
        grouped_data[docid].append(query)

    # data là một list chứa các item, mỗi item là một list chứa list các câu hỏi và 1 docid duy nhất
    data = [[query_list, docid] for docid, query_list in grouped_data.items()]

    ids, aux_ids = None, None
    # lấy prev_id là lấy bộ số đã được phân loại trước trong file k.pt.code
    if prev_id is not None:
        ids = [[0, *line] for line in json.load(open(prev_id))]
    else:
        ids = [[0]] * len(corpus)
    # đưa qua lớp BiDataset để trả về input đầu vào cho huấn luyện 
    dataset = BiDataset(data=data, corpus=corpus, tokenizer=tokenizer,
                        max_doc_len=128, max_q_len=32, ids=ids, batch_size=in_batch_size, aux_ids=aux_ids)
    accelerator.print(f'data size={len(dataset)}')

    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    optimizer = AdamW(model.parameters(), lr)
    # optimizer = get_optimizer(model, lr=lr)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    scheduler = get_constant_schedule(optimizer)

    w_1 = {'cl_loss': 0.5, 'all_cl_loss': 0, 'ce_loss': 0, 'code_loss': 0.5, 'aux_code_loss': 0, 'mse_loss': 0,
           'cl_dd_loss': 0, 'clb_loss': 0}
    w_2 = {'cl_loss': 0.5, 'all_cl_loss': 0, 'ce_loss': 0.5, 'code_loss': 0.5, 'aux_code_loss': 0, 'mse_loss': 0,
           'cl_dd_loss': 0.1, 'clb_loss': 0}
    w_3 = {'cl_loss': 0, 'all_cl_loss': 0, 'ce_loss': 0.5, 'code_loss': 1, 'aux_code_loss': 0, 'mse_loss': 0,
           'cl_dd_loss': 0, 'clb_loss': 0}
    loss_w = [None, w_1, w_2, w_3][config['loss_w']]

    step, epoch = 0, 0
    epoch_step = len(data_loader) // in_batch_size
    # safe_save(accelerator, model, save_path, -1, end_epoch=end_epoch)
    last_checkpoint = None

    for _ in range(epochs):
        accelerator.print(f'Training {save_path} {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        loss_report = []
        for batch in tk0:
            step += 1
            with accelerator.accumulate(model):
                losses = OurTrainer.train_step(model, batch, gathered=False)
                loss = sum([v * loss_w[k] for k, v in losses.items()])
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                loss = accelerator.gather(loss).mean().item()
                loss_report.append(loss)
                tk0.set_postfix(loss=sum(loss_report[-100:]) / len(loss_report[-100:]))

                if in_batch_size != 1 and step > (epoch + 1) * epoch_step:
                    epoch, last_checkpoint = safe_save(accelerator, model, save_path, epoch, end_epoch=end_epoch,
                                                       save_step=save_step,
                                                       last_checkpoint=last_checkpoint)
                if epoch >= end_epoch:
                    break
        if in_batch_size == 1:
            epoch = safe_save(accelerator, model, save_path, epoch, end_epoch=end_epoch, save_step=save_step)

    return last_checkpoint


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
    dict_idx = {}
    new_corpus = []
    count = 0
    for art in corpus[0]["articles"]: 
        # new_corpus[art['article_id']] = art['title'] + art['text']
        new_corpus.append(art['title'] + art['text'])
        dict_idx[art['article_id']] = count
        count += 1
    corpus = new_corpus
    # new_corpus = {}
    # for art in corpus[0]["articles"]: 
    #     new_corpus[art['article_id']] = art['title'] + art['text']
    # corpus = new_corpus
    new_data = []
    for item in data['items']: 
        for art in item['relevant_articles']: 
            new_data.append([
                item['question_full'],
                dict_idx[art['law_id'] + art['article_id']]
            ])
    data = new_data

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

    # seen_split = json.load(open(f'{dev_data}.seen'))
    # unseen_split = json.load(open(f'{dev_data}.unseen'))
    len_dev = len(data)
    seen_split = list(range(len_dev))
    unseen_split = list(range(len_dev))

    for epoch in range(epochs, -1, -1):
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
        print(eval_all([predictions[j] for j in seen_split], [query_ids[j] for j in seen_split]))
        print(eval_all([predictions[j] for j in unseen_split], [query_ids[j] for j in unseen_split]))


@torch.no_grad()
def our_encode(data_loader, model: Model, keys='doc'):
    collection = []
    code_collection = []
    for batch in tqdm(data_loader):
        batch = {k: v.cuda() for k, v in batch.items() if v is not None}
        output: QuantizeOutput = model(input_ids=batch[keys], attention_mask=batch[keys].ne(0),
                                       decoder_input_ids=batch['ids'],
                                       aux_ids=None, return_code=False,
                                       return_quantized_embedding=False, use_constraint=False)
        sentence_embeddings = output.continuous_embeds.cpu().tolist()
        code = output.probability.argmax(-1).cpu().tolist()
        code_collection.extend(code)
        collection.extend(sentence_embeddings)
    collection = np.array(collection, dtype=np.float32)
    return collection, code_collection

def build_index(collection, shard=True, dim=None, gpu=True):
    t = time.time()
    dim = collection.shape[1] if dim is None else dim
    cpu_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    # cpu_index = faiss.index_factory(dim, 'OPQ32,IVF1024,PQ32')
    if gpu:
        ngpus = faiss.get_num_gpus()
        co = faiss.GpuMultipleClonerOptions()
        co.shard = shard
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)
        index = gpu_index
    else:
        index = cpu_index

    # gpu_index.train(xb)
    index.add(collection)
    print(f'build index of {len(collection)} instances, time cost ={time.time() - t}')
    return index


def do_retrieval(xq, index, k=1):
    t = time.time()
    distance, rank = index.search(xq, k)
    print(f'search {len(xq)} queries, time cost ={time.time() - t}')
    return rank, distance


def do_epoch_encode(model: Model, data, corpus, ids, tokenizer, batch_size, save_path, epoch, n_code):
    corpus_q = [['', i] for i in range(len(corpus))]
    corpus_data = BiDataset(data=corpus_q, corpus=corpus, tokenizer=tokenizer, max_doc_len=128, max_q_len=32, ids=ids)
    data_loader = torch.utils.data.DataLoader(corpus_data, collate_fn=corpus_data.collate_fn, batch_size=batch_size,
                                              shuffle=False, num_workers=16)

    collection, doc_code = our_encode(data_loader, model, 'doc')
    # doc_code = [0] * len(corpus)

    print(collection.shape)
    index = build_index(collection, gpu=False)

    q_corpus = ['' for _ in range(len(corpus))]
    corpus_data = BiDataset(data=data, corpus=q_corpus, tokenizer=tokenizer, max_doc_len=128, max_q_len=32, ids=ids)
    data_loader = torch.utils.data.DataLoader(corpus_data, collate_fn=corpus_data.collate_fn, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    queries, query_code = our_encode(data_loader, model, 'query')

    rank, distance = do_retrieval(queries, index, k=100)
    rank = rank.tolist()

    json.dump(rank, open(f'{save_path}/{epoch}.pt.rank', 'w'))
    all_doc_code = [prefix[1:] + [current] for prefix, current in zip(ids, doc_code)]
    json.dump(all_doc_code, open(f'{save_path}/{epoch}.pt.code', 'w'))
    write_pkl(collection, f'{save_path}/{epoch}.pt.collection')

    print('Doc_code balance', balance(doc_code, ids, ncentroids=n_code))
    print('Doc_code conflict', conflict(doc_code, ids))

    normed_collection = norm_by_prefix(collection, ids)
    nc = n_code
    centroids, code = constrained_km(normed_collection, nc)
    print('Kmeans balance', balance(code, ids))
    print('Kmeans conflict', conflict(code, ids))
    write_pkl(centroids, f'{save_path}/{epoch}.pt.kmeans.{nc}')
    json.dump(code, open(f'{save_path}/{epoch}.pt.kmeans_code.{nc}', 'w'))

    query_ids = [x[1] for x in data]


def test_dr(config):
    model_name = config.get('model_name', 't5-base')
    code_num = config.get('code_num', 512)
    code_length = config.get('code_length', 1)
    prev_id = config.get('prev_id', None)
    save_path = config.get('save_path', None)

    dev_data = config.get('dev_data', config.get('dev_data'))
    corpus_data = config.get('corpus_data', config.get('corpus_data'))
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 128)

    data = json.load(open(dev_data))
    corpus = json.load(open(corpus_data))
    dict_idx = {}
    new_corpus = []
    count = 0
    for art in corpus[0]["articles"]: 
        # new_corpus[art['article_id']] = art['title'] + art['text']
        new_corpus.append(art['title'] + art['text'])
        dict_idx[art['article_id']] = count
        count += 1
    corpus = new_corpus
    # new_corpus = {}
    # for art in corpus[0]["articles"]: 
    #     new_corpus[art['article_id']] = art['title'] + art['text']
    # corpus = new_corpus
    new_data = []
    for item in data['items']: 
        for art in item['relevant_articles']: 
            new_data.append([
                item['question_full'],
                dict_idx[art['law_id'] + art['article_id']]
            ])
    data = new_data

    print('DR evaluation', f'{save_path}')
    t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = Model(model=t5, use_constraint=False, code_length=code_length, zero_inp=False, code_number=code_num)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.cuda()
    model.eval()

    if prev_id is not None:
        ids = [[0, *line] for line in json.load(open(prev_id))]
    else:
        ids = [[0]] * len(corpus)

    print(len(data), len(corpus))

    for epoch in range(epochs):
        if not os.path.exists(f'{save_path}/{epoch}.pt'):
            continue
        print('#' * 20)
        print(f'DR evaluation {save_path}/{epoch}.pt')
        safe_load(model, f'{save_path}/{epoch}.pt')
        do_epoch_encode(model, data, corpus, ids, tokenizer, batch_size, save_path, epoch, n_code=code_num)


def add_last(file_in, code_num, file_out):
    corpus_ids = json.load(open(file_in))
    docid_to_doc = defaultdict(list)
    new_corpus_ids = []
    for i, item in enumerate(corpus_ids):
        docid_to_doc[str(item)].append(i)
        new_corpus_ids.append(item + [len(docid_to_doc[str(item)]) % code_num])
    json.dump(new_corpus_ids, open(file_out, 'w'))
    return new_corpus_ids


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
        checkpoint = train(config)
        test_dr(config)

        config['save_path'] = args.save_path + f'-{loop + 1}'
        config['prev_model'] = checkpoint
        config['codebook_init'] = f'{checkpoint}.kmeans.{args.code_num}'
        config['epochs'] = 199   # default: 200  #should 199
        config['loss_w'] = 2
        checkpoint = train(config)
        test_dr(config)

        test(config)

    loop = args.max_length
    config['save_path'] = args.save_path + f'-{loop}-fit'
    config['code_length'] = loop + 1
    config['prev_model'] = checkpoint
    add_last(f'{checkpoint}.code', args.code_num, f'{checkpoint}.code.last')
    config['prev_id'] = f'{checkpoint}.code'
    config['epochs'] = 541     # default: 1000 # should: 1000
    config['loss_w'] = 3
    checkpoint = train(config)
    test_dr(config)
    test(config)
    print(checkpoint)


if __name__ == '__main__':
    main()
