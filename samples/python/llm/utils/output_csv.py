# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import csv
from pathlib import Path


def output_comments(result, use_case, writer):
    for key in result.keys():
        result[key] = ''
    writer.writerow(result)

    comment_list = []
    if use_case == 'text_gen' or use_case == 'code_gen':
        comment_list.append('input_size: Input token size')
        comment_list.append('output_size: Text/Code generation models: generated text token size')
        comment_list.append("infer_count: Limit the Text/Code generation models' output token size")
        comment_list.append('latency: Text/Code generation models: ms/token. Output token size / generation time')
        comment_list.append('1st_latency: Text/Code generation models: fisrt token time')
        comment_list.append('2nd_avg_latency: Text/Code generation models: other tokens (exclude first token) mean time')
        comment_list.append('result_md5: MD5 of generated text')
        comment_list.append('prompt_idx: Index of prompts')
    elif use_case == 'image_gen':
        comment_list.append("infer_count: specify the Tex2Image models' Inference(or Sampling) step size")
        comment_list.append('prompt_idx: Image Index')
    comment_list.append('pretrain_time: Total time of load model and compile model')
    comment_list.append('generation_time: Time for one interaction. (e.g. The duration of  answering one question or generating one picture)')
    comment_list.append('iteration=0: warm-up; iteration=-1: average (exclude warm-up)')
    comment_list.append(
        'max_rss_mem: max rss memory consumption; the value in -1 iteration row is the maximum value of all available RSS memory numbers in iterations.'
    )
    comment_list.append(
        'max_shared_mem: max shared memory consumption;'
        'the value in -1 iteration row is the maximum value of all available shared memory numbers in iterations.'
    )

    for comments in comment_list:
        result['iteration'] = comments
        writer.writerow(result)


def write_result(report_file, model, framework, device, use_case, iter_data_list, pretrain_time, model_precision):
    header = [
        'iteration',
        'model',
        'framework',
        'device',
        'pretrain_time(s)',
        'input_size',
        'infer_count',
        'generation_time(s)',
        'output_size',
        'latency(ms)',
        '1st_latency(ms)',
        '2nd_avg_latency(ms)',
        'precision',
        'max_rss_mem(MB)',
        'max_shared_mem(MB)',
        'prompt_idx',
        '1st_infer_latency(ms)',
        '2nd_infer_avg_latency(ms)',
        'result_md5',
    ]
    out_file = Path(report_file)

    if len(iter_data_list) > 0:
        with out_file.open('w+') as f:
            writer = csv.DictWriter(f, header)
            writer.writeheader()

            total_generation_time = 0
            total_num_tokens = 0
            total_input_size = 0
            total_infer_count = 0
            total_first_token_latency = 0
            total_other_tokens_avg_latency = 0
            total_first_token_infer_latency = 0
            total_other_tokens_infer_avg_latency = 0
            total_max_rss_mem_consumption = 0
            total_max_shared_mem_consumption = 0
            result = {}
            result['model'] = model
            result['framework'] = framework
            result['device'] = device
            result['pretrain_time(s)'] = round(pretrain_time, 5)
            result['precision'] = model_precision
            total_iters = len(iter_data_list)

            skip_iter_nums = 0
            for i in range(total_iters):
                iter_data = iter_data_list[i]
                generation_time = iter_data['generation_time']
                latency = iter_data['latency']
                first_latency = iter_data['first_token_latency']
                other_latency = iter_data['other_tokens_avg_latency']
                first_token_infer_latency = iter_data['first_token_infer_latency']
                other_token_infer_latency = iter_data['other_tokens_infer_avg_latency']
                rss_mem = iter_data['max_rss_mem_consumption']
                shared_mem = iter_data['max_shared_mem_consumption']
                result['iteration'] = str(iter_data['iteration'])
                if i > 0:
                    result['pretrain_time(s)'] = ''

                result['input_size'] = iter_data['input_size']
                result['infer_count'] = iter_data['infer_count']
                result['generation_time(s)'] = round(generation_time, 5) if generation_time != '' else generation_time
                result['output_size'] = iter_data['output_size']
                result['latency(ms)'] = round(latency, 5) if latency != '' else latency
                result['result_md5'] = iter_data['result_md5']
                result['1st_latency(ms)'] = round(first_latency, 5) if first_latency != '' else first_latency
                result['2nd_avg_latency(ms)'] = round(other_latency, 5) if other_latency != '' else other_latency
                result['1st_infer_latency(ms)'] = round(first_token_infer_latency, 5) if first_token_infer_latency != '' else first_token_infer_latency
                result['2nd_infer_avg_latency(ms)'] = round(other_token_infer_latency, 5) if other_token_infer_latency != '' else other_token_infer_latency
                result['max_rss_mem(MB)'] = round(rss_mem, 5) if rss_mem != '' else rss_mem
                result['max_shared_mem(MB)'] = round(shared_mem, 5) if shared_mem != '' else shared_mem
                result['prompt_idx'] = iter_data['prompt_idx']
                writer.writerow(result)

                # Skip the warm-up iteration
                if iter_data['iteration'] > 0:
                    if iter_data['generation_time'] != '':
                        total_generation_time += iter_data['generation_time']
                    if iter_data['output_size'] != '':
                        total_num_tokens += iter_data['output_size']
                    if iter_data['input_size'] != '':
                        total_input_size += iter_data['input_size']
                    if iter_data['first_token_latency'] != '':
                        total_first_token_latency += iter_data['first_token_latency']
                    if iter_data['other_tokens_avg_latency'] != '':
                        total_other_tokens_avg_latency += iter_data['other_tokens_avg_latency']
                    if iter_data['first_token_infer_latency'] != '':
                        total_first_token_infer_latency += iter_data['first_token_infer_latency']
                    if iter_data['other_tokens_infer_avg_latency'] != '':
                        total_other_tokens_infer_avg_latency += iter_data['other_tokens_infer_avg_latency']
                    if iter_data['infer_count'] != '':
                        total_infer_count += iter_data['infer_count']
                else:
                    skip_iter_nums = skip_iter_nums + 1
                if iter_data['max_rss_mem_consumption'] != '':
                    if iter_data['max_rss_mem_consumption'] > total_max_rss_mem_consumption:
                        total_max_rss_mem_consumption = iter_data['max_rss_mem_consumption']
                if iter_data['max_shared_mem_consumption'] != '':
                    if iter_data['max_shared_mem_consumption'] > total_max_shared_mem_consumption:
                        total_max_shared_mem_consumption = iter_data['max_shared_mem_consumption']
            total_iters -= skip_iter_nums
            if total_iters > 0:
                result['iteration'] = str('-1')
                result['pretrain_time(s)'] = ''
                if total_input_size > 0:
                    result['input_size'] = round(total_input_size / total_iters, 5)
                if total_infer_count > 0:
                    result['infer_count'] = round(total_infer_count / total_iters, 5)
                if total_generation_time > 0:
                    result['generation_time(s)'] = round(total_generation_time / total_iters, 5)
                if total_num_tokens > 0:
                    avg_per_token_time = total_generation_time * 1000 / total_num_tokens
                    result['output_size'] = round(total_num_tokens / total_iters, 5)
                    result['latency(ms)'] = round(avg_per_token_time, 5)
                else:
                    result['output_size'] = ''
                    result['latency(ms)'] = ''
                if total_first_token_latency > 0:
                    result['1st_latency(ms)'] = round(total_first_token_latency / total_iters, 5)
                if total_other_tokens_avg_latency > 0:
                    result['2nd_avg_latency(ms)'] = round(total_other_tokens_avg_latency / total_iters, 5)
                if total_first_token_infer_latency > 0:
                    result['1st_infer_latency(ms)'] = round(total_first_token_infer_latency / total_iters, 5)
                if total_other_tokens_infer_avg_latency > 0:
                    result['2nd_infer_avg_latency(ms)'] = round(total_other_tokens_infer_avg_latency / total_iters, 5)
                if total_max_rss_mem_consumption > 0:
                    result['max_rss_mem(MB)'] = total_max_rss_mem_consumption
                if total_max_shared_mem_consumption > 0:
                    result['max_shared_mem(MB)'] = total_max_shared_mem_consumption
                writer.writerow(result)

            output_comments(result, use_case, writer)
