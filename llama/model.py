# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import copy
import random
import json
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
from transformers import AutoTokenizer, AutoModel
import re

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)  # (bsz, partial_seqlen, dim)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float(), h
    
class FunctionLM(nn.Module):
    def __init__(self, base_model, tokenizer, func_dict, load_path=None, inference_mode="func_embedding"):
        super().__init__()
        self.inference_mode = inference_mode
        self.model = base_model
        self.tokenizer = tokenizer
        self.func_dict = func_dict
        self.func_list = {v: k for k, v in func_dict.items()}
        # self.func_embed = ColumnParallelLinear(
        #     base_model.params.dim, len(func_list), bias=False, init_method=lambda x: x
        # )
        self.func_embed = nn.Linear(base_model.params.dim, len(func_dict), bias=False).to("cuda")
        if load_path is not None and load_path != "None": # load func_embed weights
            embedding = torch.load(load_path)
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.to("cuda")
                embedding = {"weight": embedding}

            # truncate the embedding if necessary
            if embedding["weight"].shape[0] > len(func_dict):
                print(f"Truncated the function embedding from {embedding['weight'].shape[0]} to {len(func_dict)}")
                embedding["weight"] = embedding["weight"][:len(func_dict)]

            self.func_embed.load_state_dict(embedding)
        
        # set the basemodel to eval mode and freeze the weights
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.logits_bias = 0

    def set_bias(self, logits_bias):
        self.logits_bias = logits_bias

    def get_loss(self, raw_inputs, only_functoken=False):
        assert len(raw_inputs) == 1
        raw_inputs = raw_inputs[0]

        # inputs: starts with <bos>, ends without <eos>, (bsz, seqlen)
        # labels: starts without <bos>, ends with <eos>, (bsz, seqlen)
        with torch.no_grad():
            # prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=True) for x in raw_inputs]

            raw_input_ids = torch.tensor(self.tokenizer.encode(raw_inputs["text"], bos=True, eos=True))[:]
            labels = torch.tensor(self.tokenizer.encode(raw_inputs["text"], bos=True, eos=True))[:]

            if "tar_eq" not in raw_inputs:
                raw_inputs["tar_eq"] = ["<" + raw_inputs["api"] + ">"]

            for s, t, eq in zip(raw_inputs["start_token_idx"], raw_inputs["end_token_idx"], raw_inputs["tar_eq"]):
                
                # for different data formats
                if "[" in eq:
                    op = re.search(r"(\[.*?\])", eq).group(1)
                elif "<" in eq:
                    op = re.search(r"(<.*?>)", eq).group(1)
                    # print(op)

                if op not in self.func_dict:
                    op = op[1:-1]
                labels[s] = self.func_dict[op] + 32000
                labels[s+1: t] = -100
            
            # labels = labels[1:]
            if only_functoken:
                labels[labels < 32000] = -100
            inputs = raw_input_ids[:-1].expand(1, -1).to("cuda")
            labels = labels[1:].expand(1, -1).to("cuda")

            last_logits, h = self.model(inputs, 0) # h: (bsz, seqlen, dim)
            token_logits = self.model.output(h) # (bsz, seqlen, vocab_size)
            # print(h.device)
        
        func_logits = self.func_embed(h.float()) # (bsz, seqlen, len(func_list))
        
        concat_logits = torch.cat([token_logits, func_logits], dim=-1) # (bsz, seqlen, vocab_size + len(func_list))
        loss = F.cross_entropy(concat_logits.view(-1, concat_logits.shape[-1]), labels.view(-1), ignore_index=-100)
        # check p, r, f1 for each function
        pred = torch.argmax(concat_logits, dim=-1) # (bsz, seqlen)
        pred = pred.view(-1)
        labels = labels.view(-1)

        label_funcs = [labels == self.func_dict[op] + 32000 for op in self.func_dict.keys()]
        pred_funcs = [pred == self.func_dict[op] + 32000 for op in self.func_dict.keys()]
        label_funcs = torch.stack(label_funcs, dim=0)
        pred_funcs = torch.stack(pred_funcs, dim=0)
        
        tp = torch.sum(label_funcs * pred_funcs, dim=-1).detach().cpu().numpy()
        pred_funcs = torch.sum(pred_funcs, dim=-1).detach().cpu().numpy()
        true = torch.sum(label_funcs, dim=-1).detach().cpu().numpy()
        results = {
            "tp": tp,
            "pred": pred_funcs,
            "true": true
        }


        return loss, results
    
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_token: List[int] = [], # 29897: ), 3892: )=
        return_top: int = 0,
        disable_func: List[str] = [],
        disable_token: List[int] = [], # 29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        no_left_parens: bool = False,
        
        objs: List[str] = [],
    ) -> List[str]:
        
        bsz = len(prompts)
        # print("objs", objs)

        obj_encodings = [self.tokenizer.encode("<"+obj+">", bos=False, eos=False)[1:-1] for obj in objs]
        # print("obj encoding", obj_encodings)
        assert bsz == 1
        stop_token_substr = [torch.tensor(x).cuda().long() for x in stop_token if isinstance(x, list)]
        stop_token_single = [x for x in stop_token if isinstance(x, int)]
        
        func_list = list(self.func_dict.keys())

        # tokenize all the func in func_list
        func_tokens = [self.tokenizer.encode(x[1:-1], bos=False, eos=False) for x in func_list]
        
        generation_log = [] # (token, [(token, logits, prob)])
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        hs = []
        
        for cur_pos in range(start_pos, total_len):
            _, h = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            logits_token = self.model.output(h[:, -1, :]).float() # (bsz, vocab_size)
            logits_func = self.func_embed(h[:, -1, :].float()) # (bsz, len(func_list))
            if self.inference_mode != "func_embedding":
                logits_func = torch.zeros_like(logits_func) - 1e5
            
            if len(disable_token) > 0:
                logits_token[:, disable_token] = -1e5

            # topk: (bsz, 3)
            # print("after-topk", topk[1][0], [self.tokenizer.decode([x]) for x in topk[1][0].tolist()])

            for i, func in enumerate(disable_func):
                func_id = self.func_dict[func]
                logits_func[:, func_id] = -1e5
            logits_func += self.logits_bias
            logits = torch.cat([logits_token, logits_func], dim=-1)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if the prompt is ended
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            if return_top > 0:
                generation_log.append(
                    (next_token[0].item(), [(i.item(), logits[0, i.item()].item()) for i in torch.argsort(logits[0, :], descending=True)[:return_top]])
                )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

            if next_token[0] >= 32000 or next_token[0] in stop_token_single:
                # print("breaking!!")
                break

            if any([torch.equal(tokens[0, cur_pos - len(substr) + 1: cur_pos + 1], substr) for substr in stop_token_substr]):
                break


        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            if t[cur_pos] >= 32000:
                if no_left_parens:
                    decoded.append(self.tokenizer.decode(t[:cur_pos]) + self.func_list[t[cur_pos] - 32000])
                else:
                    if "<" in self.func_list[0]:
                        decoded.append(self.tokenizer.decode(t[:cur_pos]) + self.func_list[t[cur_pos] - 32000] + "(")
                    elif "[" in self.func_list[0]:
                        decoded.append(self.tokenizer.decode(t[:cur_pos]) + self.func_list[t[cur_pos] - 32000] + " <")
                    else:
                        raise NotImplementedError
            else:
                decoded.append(self.tokenizer.decode(t[:cur_pos + 1]))
        
        if return_top > 0:
            return decoded, generation_log
        else:
            return decoded
    
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token