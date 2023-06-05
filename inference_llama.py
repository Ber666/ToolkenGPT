# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
import re
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from tqdm import tqdm
from llama import ModelArgs, Transformer, Tokenizer, LLaMA, FunctionLM
from collections import Counter

from funchub.math import *


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, func_load_path: str, func_dict: dict) -> FunctionLM:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # print(checkpoints)
    assert (
        world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=1, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args).cuda().half()
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    size = ckpt_dir.split("/")[-1]
    funcmodel = FunctionLM(model, tokenizer, func_dict = func_dict, load_path=func_load_path)
    # generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return funcmodel

def func_embedding_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, return_top=5):
    cur_generation = ""
    cur_generation_with_func = ""
    start_length = []
    end_length = []

    logs = []

    funcmodel.inference_mode = "func_embedding"

    # get func list
    func_map = list(funcmodel.func_dict.keys())

    try:
        results = []
        func_calls = []
        while True:

            prompt = templates["general"].replace("[QUESTION]", question) + cur_generation
            results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=[13], return_top=return_top)
            if return_top > 0:
                results, token_log = results
                logs.append(token_log)
            endflag = True

            current_token = 0
            
            record_tokens = token_log[-1]
            # assert prompt in results[0]
            cur_generation = results[0].replace(templates["general"].replace("[QUESTION]", question), "")
            
            for op in func_map:
                
                if cur_generation.endswith(op+"("):
                    endflag = False

                    if start_length and end_length:
                        bias = 0
                        # copy the current generation to cur_generation_with_func
                        cur_generation_with_func = cur_generation
                        for i in range(len(start_length)):
                            cur_generation_with_func = cur_generation_with_func[:start_length[i]+bias] +func_calls[i] + cur_generation_with_func[end_length[i]+bias:]
                            bias += len(func_calls[i]) - (end_length[i] - start_length[i])
                    else:
                        cur_generation_with_func = cur_generation

                    prompt = templates[op].replace("[QUESTION]", question) + cur_generation_with_func
                    len_prompt = len(prompt)

                    funcmodel.inference_mode = "baseline"

                    results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=[29897, 3892], return_top=return_top)

                    funcmodel.inference_mode = "func_embedding"
                
                    if return_top > 0:
                        results, token_log = results
                        logs.append(token_log)
                    # logs.append(token_log)
                    # assert prompt in results[0]

                    generated = results[0][len_prompt:]
                    cur_generation += generated
                    
                    args = cur_generation.split(op)[-1].replace("=", "").replace(">", "").replace("((", "(").replace("))", ")") # shouldn't have >, but there is one in case study
                    # remove any $ in the args
                    args = args.replace("$", "")

                    # remove , in the args
                    if ", " in args:
                        args = args.replace(", ", ";").replace(",", "").replace(";", ", ")

                    args = args.replace(" ", "")

                    if "(" not in args or ")" not in args:
                        raise Exception("invalid args")

                    # handle %
                    if '%' in args:
                        temp = args.split("(")[1].split(")")[0].split(",")

                        for arg_i, arg in enumerate(temp):
                            # if have percentage, convert to decimal
                            if "%" in arg:
                                arg = arg.replace("%", "").strip()
                                arg = str(float(arg) / 100)
                            
                            temp[arg_i] = arg
                        
                        args = f"({', '.join(temp)})"
                    
                    try:
                        res = eval(f"{op[1:-1]}_{args}")
                        func_calls.append(f"{op}{args} = {res}")

                        start_length.append(len(cur_generation.split(op)[0]))

                        cur_generation = cur_generation.split(op)[0] + str(res)

                        end_length.append(len(cur_generation))

                        # only generate the next token
                        # disable all the numbers
                        prompt = templates["general"].replace("[QUESTION]", question) + cur_generation
                        results = funcmodel.generate([prompt], max_gen_len=1, temperature=temperature, top_p=top_p, stop_token=[13], return_top=return_top,
                                                     disable_token = [29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929]) # disable all the numbers: 0-9
                        if return_top > 0:
                            results, token_log = results
                            logs.append(token_log)

                        
                        cur_generation = results[0].replace(templates["general"].replace("[QUESTION]", question), "")

                    except:
                        # backtrace 
                        current_token += 1
                        decode_token = lambda x: funcmodel.tokenizer.decode(x) if x < 32000 else func_map[x - 32000]
                        cur_generation = cur_generation.split(op)[0] + decode_token(record_tokens[1][current_token][0])
                    
                    break
            if endflag:
                break


        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            # need to return logs
            # "token_log": logs,
            "status": "success"
        }
            # f.write(json.dumps(log) + "\n")

    except Exception as e:
        # if local_rank == 0:
        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": str(e)
        }
    return log

def baseline_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len):
    funcmodel.inference_mode = "baseline"
    cur_generation = ""
    try:
        prompt = templates["general"].replace("[QUESTION]", question) + cur_generation
        results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=[13])
        
        cur_generation = results[0].replace(templates["general"].replace("[QUESTION]", question), "")

        log = {
            "case_idx": case_idx,
            "question": question,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": "success"
        }
            # f.write(json.dumps(log) + "\n")

    except Exception as e:
        # if local_rank == 0:
        log = {
            "case_idx": case_idx,
            "question": question,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": str(e)
        }

    return log

def main(ckpt_dir: str, tokenizer_path: str, temperature: float = 0, top_p: float = 0.95, mode: str = "baseline", dataset = "original", return_top: int = 5, logits_bias: float = 0, func_load_path: str = "None", st_idx=0, ed_idx=10000):
    # set random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1)
    np.random.seed(1)

    size = ckpt_dir.split("/")[-1]

    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')
    
    templates = {}
    

    if dataset == "gsm8k-xl":
        for name in os.listdir("data/gsm8k-xl/template"):
            with open(f"data/gsm8k-xl/template/{name}") as f:
                templates[name.split("_")[-1].replace(".txt", "")] = f.read()
        with open(f"data/gsm8k-xl/test.json") as f:
            data = [json.loads(line) for line in f.readlines()]
            
        raw_test_cases = [i["question"] for i in data]
        enhanced_v = [i["enhanced_v"] for i in data]
        
        test_cases = []
        for v, q in zip(enhanced_v, raw_test_cases):
            # parse {v_1}, {v_2}, ... in q and fill with v
            for i in range(len(v)):
                q = q.replace(f"{{v_{i+1}}}", str(v[i]))

            test_cases.append(q)

        max_gen_len = 512
        func_dict = {
            "<add>": 0,
            "<subtract>": 1,
            "<multiply>": 2,
            "<divide>": 3,
            }
    elif dataset == "funcqa_mh":
        for name in os.listdir("data/funcqa/template_mh"):
            with open(f"data/funcqa/template_mh/{name}") as f:
                templates[name.split("_")[-1].replace(".txt", "")] = f.read()

        with open("data/funcqa/funcqa_mh.json") as f:
            data = json.load(f)
    
        test_cases = [i["question"] for i in data]

        max_gen_len = 512

        func_dict = {
            "<add>": 0,
            "<subtract>": 1,
            "<multiply>": 2,
            "<divide>": 3,
            "<power>": 4,
            "<sqrt>": 5,
            "<log>": 6,
            "<ln>": 7,
            "<lcm>": 8,
            "<gcd>": 9,
            "<remainder>": 10,
            "<choose>": 11,
            "<permutate>": 12
        }
    
    elif dataset == "funcqa_oh":
        for name in os.listdir("data/funcqa/template_oh"):
            with open(f"data/ohqa/template_oh/{name}") as f:
                templates[name.split("_")[-1].replace(".txt", "")] = f.read()
        
        with open("data/funcqa/funcqa_oh.json") as f:
            data = json.load(f)
        
        max_gen_len = 512
        
        func_dict = {
            "<add>": 0,
            "<subtract>": 1,
            "<multiply>": 2,
            "<divide>": 3,
            "<power>": 4,
            "<sqrt>": 5,
            "<log>": 6,
            "<ln>": 7,
            "<lcm>": 8,
            "<gcd>": 9,
            "<remainder>": 10,
            "<choose>": 11,
            "<permutate>": 12
        }
        
        test_cases = [i["question"] for i in data]

    funcmodel = load(ckpt_dir, tokenizer_path, local_rank, world_size, func_load_path=func_load_path, func_dict=func_dict)
    funcmodel.set_bias(logits_bias)
    funcmodel.eval()

    for case_idx, question in tqdm(enumerate(test_cases), total=len(test_cases)):
        if case_idx < st_idx:
            continue
        if case_idx >= ed_idx:
            break
        if mode == "func_embedding":
            log = func_embedding_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, return_top)
        elif mode == "baseline":
            log = baseline_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len)
        
        if local_rank == 0:
            try:
                func_model_name = func_load_path.split('/')[-1].split('.')[0]
            except:
                func_model_name = func_load_path

            output_dir = f"outputs/{dataset}"
            os.makedirs(output_dir, exist_ok=True)

            with open(f"{output_dir}/inference-{size}-{func_model_name}-{mode}-{dataset}-{logits_bias}.jsonl", "a") as f:
                f.write(json.dumps(log) + "\n")


if __name__ == "__main__":
    fire.Fire(main)