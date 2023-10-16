# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging as log


def print_metrics(iter_num, iter_data, tms=None, tms_infer=None, generated=None, warm_up=False, max_rss_mem=-1, max_shared_mem=-1):
    if tms is None:
        tms = []
    if tms_infer is None:
        tms_infer = []
    iter_str = str(iter_num)
    if warm_up:
        iter_str = 'warm-up'
    if iter_data['input_size'] != '':
        log.info(f"[{iter_str}] Input token size: {iter_data['input_size']}")
    if iter_data['output_size'] != '':
        log.info(f"[{iter_str}] Output size: {iter_data['output_size']}")
    if iter_data['infer_count'] != '':
        log.info(f"[{iter_str}] Infer count: {iter_data['infer_count']}")
    if iter_data['generation_time'] != '':
        log.info(f"[{iter_str}] Generation Time: {iter_data['generation_time']:.2f}s")
    if iter_data['latency'] != '':
        log.info(f"[{iter_str}] Latency: {iter_data['latency']:.2f} ms/token")
    if generated is not None:
        log.info(f'[{iter_str}] Generated:\n{generated}')
    if iter_data['result_md5'] != '':
        log.info(f"[{iter_str}] Result MD5:{iter_data['result_md5']}")
    if len(tms) > 0:
        iter_data['first_token_latency'] = tms[0] * 1000 if len(tms) > 0 else -1
        iter_data['other_tokens_avg_latency'] = sum(tms[1:]) / (len(tms)-1) * 1000 if len(tms) > 1 else -1
        iter_data['first_token_infer_latency'] = tms_infer[0] * 1000 if len(tms_infer) > 0 else -1
        iter_data['other_tokens_infer_avg_latency'] = sum(tms_infer[1:]) / (len(tms_infer) - 1) * 1000 if len(tms_infer) > 1 else -1
        log.info(f"[{iter_str}] First token time: {iter_data['first_token_latency']:.2f} ms,"
                 + f" other tokens average time: {iter_data['other_tokens_avg_latency']:.2f} ms/token,"
                 + f" len of tokens: {len(tms)}")
        log.info(f"[{iter_str}] First token infer time: {iter_data['first_token_infer_latency']:.2f} ms,"
                 + f" other tokens average infer time: {iter_data['other_tokens_infer_avg_latency']:.2f} ms/token,"
                 + f" len of tokens: {len(tms)}")
    if max_rss_mem != '' and max_rss_mem > -1:
        log.info(f'[{iter_str}] max rss memory cost:\n{max_rss_mem}')
    if max_shared_mem != '' and max_shared_mem > -1:
        log.info(f'[{iter_str}] max shared memory cost:\n{max_shared_mem}')


def print_average(iter_data_list):
    if (len(iter_data_list) <= 1):
        # 1st iteration is the warm-up iteration
        return
    total_generation_time = 0
    total_num_tokens = 0
    warm_up_iters = 0
    for iter_data in iter_data_list:
        if iter_data['iteration'] == 0:
            # Exclude the warm-up iteration
            warm_up_iters = warm_up_iters + 1
            continue
        if iter_data['generation_time'] != '':
            total_generation_time += iter_data['generation_time']
        if iter_data['output_size'] != '':
            total_num_tokens += iter_data['output_size']

    total_iters = len(iter_data_list) - warm_up_iters

    if total_iters > 0:
        log.info('<<< Warm-up iteration is excluded. >>>')
        log.info(f'[Total] Iterations: {total_iters}')
        if total_num_tokens > 0:
            log.info(f'[Total] Output size: {total_num_tokens} tokens')
        if total_generation_time > 0:
            avg_per_iter_time = total_generation_time / total_iters
            log.info(f'[Average] Iteration time: {avg_per_iter_time:.2f}s')
            if total_num_tokens > 0:
                avg_per_token_time = total_generation_time * 1000 / total_num_tokens
                log.info(f'[Average] Latency: {avg_per_token_time:.2f} ms/token')
