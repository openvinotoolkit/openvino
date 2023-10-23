# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import json
import logging as log
from pathlib import Path
from utils.config_class import DEFAULT_MODEL_CLASSES, USE_CASES, OV_MODEL_CLASSES_MAPPING, PT_MODEL_CLASSES_MAPPING


def get_prompts(args):
    prompts_list = []
    if args['prompt'] is None and args['prompt_file'] is None:
        if args['use_case'] == 'text_gen':
            prompts_list.append('What is OpenVINO?')
        elif args['use_case'] == 'code_gen':
            prompts_list.append('def print_hello_world():')
        elif args['use_case'] == 'image_gen':
            prompts_list.append('sailing ship in storm by Leonardo da Vinci')
    elif args['prompt'] is not None and args['prompt_file'] is not None:
        raise RuntimeError('== prompt and prompt file should not exist together ==')
    else:
        if args['prompt'] is not None:
            prompts_list.append(args['prompt'])
        else:
            input_prompt = args['prompt_file']
            if input_prompt.endswith('.jsonl'):
                if os.path.exists(input_prompt):
                    log.info(f'Read prompts from {input_prompt}')
                    with open(input_prompt, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line)
                            prompts_list.append(data['prompt'])
                else:
                    raise RuntimeError('== The prompt file does not exist ==')
            else:
                prompts_list.append(input_prompt)
    return prompts_list


def set_default_param_for_ov_config(ov_config):
    if 'PERFORMANCE_HINT' not in ov_config:
        ov_config['PERFORMANCE_HINT'] = 'LATENCY'
    # With this PR https://github.com/huggingface/optimum-intel/pull/362, we are able to disable model cache
    if 'CACHE_DIR' not in ov_config:
        ov_config['CACHE_DIR'] = ''
    # OpenVINO self have default value 2 for nstreams on machine with 2 nodes. Reducing memory consumed via changing nstreams to 1.
    if 'NUM_STREAMS' not in ov_config:
        ov_config['NUM_STREAMS'] = '1'


def analyze_args(args):
    model_args = {}
    model_args['prompt'] = args.prompt
    model_args['prompt_file'] = args.prompt_file
    model_args['model_id'] = args.model_id
    model_args['infer_count'] = args.infer_count
    model_args['images'] = args.images
    model_args['seed'] = args.seed
    model_args['mem_consumption'] = args.memory_consumption
    model_args['batch_size'] = args.batch_size
    model_args['fuse_decoding_strategy'] = args.fuse_decoding_strategy
    model_args['make_stateful'] = args.make_stateful
    model_args['save_prepared_model'] = args.save_prepared_model
    model_args['num_beams'] = args.num_beams
    model_args['fuse_cache_reorder'] = args.fuse_cache_reorder
    model_args['torch_compile_backend'] = args.torch_compile_backend

    model_path = args.model
    model_framework = args.framework
    model_path = Path(model_path)
    if not model_path.exists():
        raise RuntimeError(f'==Failure FOUND==: Incorrect model path:{model_path}')
    if model_framework == 'ov':
        model_names = args.model.split('/')
        use_case, model_name = get_use_case(model_names)
    elif model_framework == 'pt':
        if len(model_args['model_id']) != 0:
            model_args['model_id'] = args.model_id
        model_names = args.model.split('/')
        use_case, model_name = get_use_case(model_names)
    model_args['use_case'] = use_case
    if use_case == 'code_gen' and not model_args['prompt'] and not model_args['prompt_file']:
        model_args['prompt'] = 'def print_hello_world():'
    model_args['config'] = {}
    if args.load_config is not None:
        config = get_config(args.load_config)
        if type(config) is dict and len(config) > 0:
            model_args['config'] = config
    if model_framework == 'ov':
        set_default_param_for_ov_config(model_args['config'])
        log.info(f"ov_config={model_args['config']}")
    elif model_framework == 'pt':
        log.info(f"pt_config={model_args['config']}")
    model_args['model_type'] = get_model_type(model_name, use_case, model_framework)
    model_args['model_name'] = model_name
    return model_path, model_framework, model_args, model_name


def get_use_case(model_name_list):
    for model_name in model_name_list:
        for case, model_ids in USE_CASES.items():
            for model_id in model_ids:
                if model_id in model_name:
                    log.info(f'==SUCCESS FOUND==: use_case: {case}, model_name: {model_name}')
                    return case, model_name
    raise RuntimeError('==Failure FOUND==: no use_case found')


def get_config(config):
    with open(config, 'r') as f:
        try:
            ov_config = json.load(f)
        except Exception:
            raise RuntimeError(f'==Parse file:{config} failiure, json format is incorrect ==')
    return ov_config


def get_model_type(model_name, use_case, model_framework):
    default_model_type = DEFAULT_MODEL_CLASSES.get(use_case)
    if model_framework == 'ov':
        for cls in OV_MODEL_CLASSES_MAPPING:
            if cls in model_name:
                return cls
    elif model_framework == 'pt':
        for cls in PT_MODEL_CLASSES_MAPPING:
            if cls in model_name:
                return cls
    return default_model_type


def get_ir_conversion_frontend(cur_model_name, model_name_list):
    ir_conversion_frontend = ''
    idx = 0
    for model_name in model_name_list:
        # idx+1 < len(model_name_list) to avoid out of bounds index of model_name_list
        if model_name == cur_model_name and idx + 1 < len(model_name_list):
            ir_conversion_frontend = model_name_list[idx + 1]
            break
        idx = idx + 1
    return ir_conversion_frontend


def get_model_precision(model_name_list):
    precision_list = ['FP32', 'FP16', 'FP16-INT8', 'INT8', 'INT8_compressed_weights', 'INT8_quantized', 'PT_compressed_weights']
    precision_list += ['OV_FP32-INT8', 'OV_FP16-INT8', 'PT_FP32-INT8', 'PT_FP16-INT8']
    model_precision = 'unknown'
    # Search from right to left of model path
    for i in range(len(model_name_list) - 1, -1, -1):
        for precision in precision_list:
            if model_name_list[i] == precision:
                model_precision = precision
                break
        if model_precision != 'unknown':
            break
    return model_precision
