# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import torch
import timm
from utils.config_class import PT_MODEL_CLASSES_MAPPING, TOKENIZE_CLASSES_MAPPING, DEFAULT_MODEL_CLASSES
import os
import time
import logging as log

MAX_CONNECT_TIME = 50


def set_bf16(model, device, **kwargs):
    try:
        if len(kwargs['config']) > 0 and kwargs['config'].get('PREC_BF16') and kwargs['config']['PREC_BF16'] is True:
            model = model.to(device.lower(), dtype=torch.bfloat16)
            log.info('Set inference precision to bf16')
    except Exception:
        log.error('Catch exception for setting inference precision to bf16.')
        raise RuntimeError('Set prec_bf16 fail.')
    return model


def run_torch_compile(model, backend='openvino'):
    if backend == 'pytorch':
        log.info('Running torch.compile() with pytorch backend')
        start = time.perf_counter()
        compiled_model = torch.compile(model)
        end = time.perf_counter()
        compile_time = end - start
        log.info(f'Compiling model via torch.compile() took: {compile_time}')
    else:
        log.info('Running torch.compile() with openvino backend')
        start = time.perf_counter()
        compiled_model = torch.compile(model, backend=backend)
        end = time.perf_counter()
        compile_time = end - start
        log.info(f'Compiling model via torch.compile() took: {compile_time}')
    return compiled_model


def get_text_model_from_huggingface(model_path, connect_times, **kwargs):
    model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
    model_class = PT_MODEL_CLASSES_MAPPING[model_type]
    token_class = TOKENIZE_CLASSES_MAPPING[model_type]
    from_pretrain_time = 0
    try:
        start = time.perf_counter()
        tokenizer = token_class.from_pretrained(kwargs['model_id'])
        model = model_class.from_pretrained(kwargs['model_id'])
        end = time.perf_counter()
        from_pretrain_time = end - start
        log.info('Get tokenizer and model from huggingface success')
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
    except Exception:
        log.info('Try to connect huggingface times: {connect_times}....')
        if connect_times > MAX_CONNECT_TIME:
            raise RuntimeError(f'==Failure ==: connect times {MAX_CONNECT_TIME}, connect huggingface failed')
        time.sleep(3)
        connect_times += 1
        model, tokenizer, from_pretrain_time = get_text_model_from_huggingface(model_path, connect_times, **kwargs)
    return model, tokenizer, from_pretrain_time


def create_text_gen_model(model_path, device, **kwargs):
    model_path = Path(model_path)
    from_pretrain_time = 0
    if model_path.exists():
        if model_path.is_dir():
            # Checking if the list is empty or not
            if len(os.listdir(model_path)) == 0:
                if kwargs['model_id'] != '':
                    log.info('Get text model from huggingface...')
                    connect_times = 1
                    model, tokenizer, from_pretrain_time = get_text_model_from_huggingface(model_path, connect_times, **kwargs)
                else:
                    raise RuntimeError('==Failure ==: the model id of huggingface should not be empty!')
            else:
                log.info(f'Load text model from model path:{model_path}')
                default_model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
                model_type = kwargs.get('model_type', default_model_type)
                model_class = PT_MODEL_CLASSES_MAPPING.get(model_type, PT_MODEL_CLASSES_MAPPING[default_model_type])
                token_class = TOKENIZE_CLASSES_MAPPING.get(model_type, TOKENIZE_CLASSES_MAPPING[default_model_type])
                start = time.perf_counter()
                if model_type == 'chatglm':
                    model = model_class.from_pretrained(model_path, trust_remote_code=True).to('cpu', dtype=float)
                else:
                    model = model_class.from_pretrained(model_path, trust_remote_code=True)
                tokenizer = token_class.from_pretrained(model_path, trust_remote_code=True)
                end = time.perf_counter()
                from_pretrain_time = end - start
        else:
            raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory')
    else:
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')

    log.info(f'model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    if device is not None:
        gptjfclm = 'transformers.models.gptj.modeling_gptj.GPTJForCausalLM'
        lfclm = 'transformers.models.llama.modeling_llama.LlamaForCausalLM'
        bfclm = 'transformers.models.bloom.modeling_bloom.BloomForCausalLM'
        gpt2lmhm = 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'
        gptneoxclm = 'transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM'
        chatglmfcg = 'transformers_modules.pytorch_original.modeling_chatglm.ChatGLMForConditionalGeneration'
        real_base_model_name = str(type(model)).lower()
        log.info('Real base model=', real_base_model_name)
        # bfclm will trigger generate crash.

        # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
        if device.upper() == 'GPU':
            device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
        else:
            device = torch.device(device.lower())
        log.info(f'Torch device was set to: {device}')

        if any(x in real_base_model_name for x in [gptjfclm, lfclm, bfclm, gpt2lmhm, gptneoxclm, chatglmfcg]):
            model = set_bf16(model, device, **kwargs)
        else:
            if len(kwargs['config']) > 0 and kwargs['config'].get('PREC_BF16') and kwargs['config']['PREC_BF16'] is True:
                log.info('Param [bf16/prec_bf16] will not work.')
            model.to(device.lower())
    else:
        raise RuntimeError('==Failure ==: no device to load')

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        compiled_model = run_torch_compile(model, backend)
        model = compiled_model
    return model, tokenizer, from_pretrain_time


def get_image_model_from_huggingface(model_path, connect_times, **kwargs):
    model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
    model_class = PT_MODEL_CLASSES_MAPPING[model_type]
    from_pretrain_time = 0
    try:
        start = time.perf_counter()
        pipe = model_class.from_pretrained(kwargs['model_id'])
        pipe.save_pretrained(model_path)
        end = time.perf_counter()
        from_pretrain_time = end - start
        log.info('Get image model from huggingface success')
    except Exception:
        log.info(f'Try to connect huggingface times: {connect_times}....')
        if connect_times > MAX_CONNECT_TIME:
            raise RuntimeError(f'==Failure ==: connect times {MAX_CONNECT_TIME}, connect huggingface failed')
        time.sleep(3)
        connect_times += 1
        pipe, from_pretrain_time = get_image_model_from_huggingface(model_path, connect_times, **kwargs)
    return pipe, from_pretrain_time


def create_image_gen_model(model_path, device, **kwargs):
    model_path = Path(model_path)
    from_pretrain_time = 0
    if model_path.exists():
        if model_path.is_dir():
            # Checking if the list is empty or not
            if len(os.listdir(model_path)) == 0:
                if kwargs['model_id'] != '':
                    log.info('Get image model from huggingface...')
                    connect_times = 1
                    pipe, from_pretrain_time = get_image_model_from_huggingface(model_path, connect_times, **kwargs)
                else:
                    raise RuntimeError('==Failure ==: the model id of huggingface should not be empty!')
            else:
                log.info(f'Load image model from model path:{model_path}')
                model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
                model_class = PT_MODEL_CLASSES_MAPPING[model_type]
                start = time.perf_counter()
                pipe = model_class.from_pretrained(model_path)
                end = time.perf_counter()
                from_pretrain_time = end - start
        else:
            raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory')
    else:
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')

    log.info(f'Model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    if device:
        # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
        if device.upper() == 'GPU':
            device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
        else:
            device = torch.device(device.lower())
        log.info(f'Torch device was set to: {device}')

        pipe.to(device)
    else:
        raise RuntimeError('==Failure ==: no device to load')

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        compiled_model = run_torch_compile(pipe, backend)
        pipe = compiled_model
    return pipe, from_pretrain_time


def create_image_classification_model(model_path, device, **kwargs):
    model_path = Path(model_path)
    model_file = None
    model_id = None
    if model_path.exists():
        if model_path.is_dir():
            model_file = list(model_path.glob('*.pth'))
            if model_file:
                model_file = model_file[0]
            else:
                model_file = None
            model_id = model_path.name
        else:
            model_file = model_path
            model_id = model_path.name.replace('.pth', '')
    else:
        model_id = model_path.name.replace('.pth', '')
    model = timm.create_model(model_id, pretrained=model_file is None)
    if model_file:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
    else:
        log.info(model.state_dict())
        torch.save(model.state_dict(), model_path / f'{model_id}.pth')
    if device:
        # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
        if device.upper() == 'GPU':
            device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
        else:
            device = torch.device(device.lower())
        log.info(f'Torch device was set to: {device}')

        model.to(device)
    else:
        raise RuntimeError('==Failure ==: no device to load')
    model.eval()
    data_config = timm.data.resolve_data_config([], model=model_id, use_test_size=True)
    input_size = (1,) + data_config['input_size']

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        compiled_model = run_torch_compile(model, backend)
        model = compiled_model
    return model, input_size


def create_ldm_super_resolution_model(model_path, device, **kwargs):
    model_path = Path(model_path)
    from_pretrain_time = 0
    if model_path.exists():
        if model_path.is_dir():
            # Checking if the list is empty or not
            if len(os.listdir(model_path)) == 0:
                if kwargs['model_id'] != '':
                    log.info('Get super resolution model from huggingface...')
                    connect_times = 1
                    pipe, from_pretrain_time = get_image_model_from_huggingface(model_path, connect_times, **kwargs)
                else:
                    raise RuntimeError('==Failure ==: the model id of huggingface should not be empty!')
            else:
                log.info(f'Load image model from model path:{model_path}')
                model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
                model_class = PT_MODEL_CLASSES_MAPPING[model_type]
                start = time.perf_counter()
                pipe = model_class.from_pretrained(model_path)
                end = time.perf_counter()
                from_pretrain_time = end - start
        else:
            raise RuntimeError(f'==Failure ==: model path:{model_path} is not directory')
    else:
        raise RuntimeError(f'==Failure ==: model path:{model_path} is not exist')

    log.info(f'Model path:{model_path}, from pretrained time: {from_pretrain_time:.2f}s')

    if device:
        # If the device is set to GPU there's a need to substitute it with 'cuda' so it will be accepted by PyTorch
        if device.upper() == 'GPU':
            device = torch.device('cuda') if torch.cuda.is_available() else log.info('CUDA device is unavailable')
        else:
            device = torch.device(device.lower())
        log.info(f'Torch device was set to: {device}')

        pipe.to(device)
    else:
        raise RuntimeError('==Failure ==: no device to load')

    if kwargs['torch_compile_backend']:
        backend = kwargs['torch_compile_backend']
        compiled_model = run_torch_compile(pipe, backend)
        pipe = compiled_model
    return pipe, from_pretrain_time
