def func_embedding_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, return_top=5):
    cur_generation = ""
    cur_generation_with_func = ""
    start_length = []
    end_length = []
    logs = []
    funcmodel.inference_mode = "func_embedding"
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
                    generated = results[0][len_prompt:]
                    cur_generation += generated
                    args = cur_generation.split(op)[-1].replace("=", "").replace(">", "").replace("((", "(").replace("))", ")")
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
                        results = funcmodel.generate([prompt], max_gen_len=1, temperature=temperature, top_p=top_p, stop_token=[13], return_top=return_top, disable_token = [29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929]) # disable all the numbers: 0-9
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
            "status": "success"
        }

    except Exception as e:
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
    except Exception as e:
        # if local_rank == 0:
        log = {
            "case_idx": case_idx,
            "question": question,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": str(e)
        }

    return log