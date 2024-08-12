from abc import ABC
from transformers.generation import GenerationMixin
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import torch
from my_prepare_data import QuantizeOutput
from my_sinkhorn import sinkhorn_raw


class Model(nn.Module, GenerationMixin, ABC):
    def __init__(self, model, use_constraint: bool, sk_epsilon: float = 0.03, sk_iters: int = 100, code_length=1,
                 zero_inp=False, code_number=10):
        super().__init__()
        self.model = model
        self.config = model.config
        self.generation_config = model.generation_config
        # self.base_model_prefix = 't5'
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