# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import argparse
import time
from pathlib import Path
import logging as log
import utils.ov_utils
import utils.pt_utils
import utils.model_utils
import torch
import numpy as np
from openvino.runtime import get_version
from utils.config_class import DEFAULT_MODEL_CLASSES
import PIL
import hashlib
import utils.metrics_print
import utils.output_csv
import utils.hook_transformers
import traceback
from transformers import set_seed
from PIL import Image
from utils.memory_profile import MemConsumption

HOOK_UTILS = {'pt': utils.hook_transformers, 'ov': utils.hook_transformers}
FW_UTILS = {'pt': utils.pt_utils, 'ov': utils.ov_utils}

DEFAULT_INFERENCE_STEPS = 20
DEFAULT_SUPER_RESOLUTION_STEPS = 50
DEFAULT_OUTPUT_TOKEN_SIZE = 512
MAX_OUTPUT_TOKEN_SIZE = 64 * 1024

mem_consumption = MemConsumption()


def gen_iterate_data(
    iter_idx='',
    in_size='',
    infer_count='',
    out_size='',
    gen_time='',
    latency='',
    res_md5='',
    max_rss_mem='',
    max_shared_mem='',
    prompt_idx='',
):
    iter_data = {}
    iter_data['iteration'] = iter_idx
    iter_data['input_size'] = in_size
    iter_data['infer_count'] = infer_count
    iter_data['output_size'] = out_size
    iter_data['generation_time'] = gen_time
    iter_data['latency'] = latency
    iter_data['result_md5'] = res_md5
    iter_data['first_token_latency'] = ''
    iter_data['other_tokens_avg_latency'] = ''
    iter_data['first_token_infer_latency'] = ''
    iter_data['other_tokens_infer_avg_latency'] = ''
    iter_data['max_rss_mem_consumption'] = max_rss_mem
    iter_data['max_shared_mem_consumption'] = max_shared_mem
    iter_data['prompt_idx'] = prompt_idx
    return iter_data


def run_text_generation(input_text, num, model, tokenizer, args, iter_data_list, prompt_index, bench_hook):
    set_seed(args['seed'])
    input_text_list = [input_text] * args['batch_size']
    log.info(f'input_text={input_text}')
    input_data = tokenizer(input_text_list, return_tensors='pt')
    input_data.pop('token_type_ids', None)
    # Remove `token_type_ids` from inputs
    input_tokens = input_data['input_ids'] if 'input_ids' in input_data else input_data
    input_token_size = input_tokens[0].numel()

    max_output_token_size = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    max_output_token_size = MAX_OUTPUT_TOKEN_SIZE if max_output_token_size > MAX_OUTPUT_TOKEN_SIZE else max_output_token_size
    if args['batch_size'] > 1:
        log.info(f"batch_size={args['batch_size']}")
        log.info(f"All input token size after padding:{input_token_size} * {args['batch_size']}")
        log.info(f"All max_output_token_size:{max_output_token_size} * {args['batch_size']}")
    else:
        log.info(f'Input token size:{input_token_size}, max_output_token_size:{max_output_token_size}')

    max_rss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    start = time.perf_counter()
    result = model.generate(**input_data, max_new_tokens=int(max_output_token_size), num_beams=args['num_beams'], use_cache=True)
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()

    generation_time = end - start
    generated_text = tokenizer.batch_decode(result)
    # Only text_gen need to minus length of input_data, because generated_text may include input_text
    num_tokens = 0
    result_md5_list = []
    for i in range(args['batch_size']):
        if 'sum' not in args['model_name'] and result[i][:input_token_size].equal(input_tokens[i]):
            generated_text_len = len(result[i]) - input_tokens[i].numel()
        else:
            generated_text_len = len(result[i])
        num_tokens += generated_text_len
        if generated_text_len > max_output_token_size:
            log.error('Output token size is over max output token size!')
        result_text = generated_text[i]
        result_md5_list.append(hashlib.md5(result_text.encode()).hexdigest())
    per_token_time = generation_time * 1000 / num_tokens
    iter_data = gen_iterate_data(
        num,
        input_token_size * args['batch_size'],
        max_output_token_size * args['batch_size'],
        num_tokens,
        generation_time,
        per_token_time,
        result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        prompt_idx=prompt_index,
    )
    iter_data_list.append(iter_data)
    tm_list = bench_hook.get_time_list()
    tm_infer_list = bench_hook.get_time_infer_list()
    utils.metrics_print.print_metrics(
        num,
        iter_data,
        tm_list,
        tm_infer_list,
        generated=generated_text[0],
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
    )
    bench_hook.clear_time_list()
    bench_hook.clear_time_infer_list()


def run_text_generation_benchmark(model_path, framework, device, args, num_iters):
    bench_hook = HOOK_UTILS[framework].BenchHook()
    model, tokenizer, pretrain_time = FW_UTILS[framework].create_text_gen_model(model_path, device, **args)
    # Override forward for statistic each forward time.
    default_model_type = DEFAULT_MODEL_CLASSES[args['use_case']]
    model_type = args.get('model_type', default_model_type)
    bench_hook.new_forward(model, model_type)

    iter_data_list = []
    input_text_list = utils.model_utils.get_prompts(args)
    if len(input_text_list) == 0:
        raise RuntimeError('==Failure prompts is empty ==')

    log.info(f'num_iters={num_iters}, num_text_list={len(input_text_list)}')
    # if num_iters == 0, just output warm-up data
    for num in range(num_iters + 1):
        prompt_idx = 0
        for input_text in input_text_list:
            run_text_generation(input_text, num, model, tokenizer, args, iter_data_list, prompt_idx, bench_hook)
            prompt_idx = prompt_idx + 1

    utils.metrics_print.print_average(iter_data_list)

    return iter_data_list, pretrain_time


def run_image_generation(input_text, nsteps, num, image_id, pipe, args, iter_data_list):
    set_seed(args['seed'])
    log.info(f'batch_size={args["batch_size"]}')
    result_md5_list = []
    max_rss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    start = time.perf_counter()
    res = pipe([input_text] * args['batch_size'], num_inference_steps=nsteps, height=512, width=512).images
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()
    for i in range(args['batch_size']):
        if num == 0:
            rslt_img_fn = args['model_name'] + '_bs' + str(args['batch_size']) + '-' + str(i + 1) + '_img_warm-up.png'
        else:
            rslt_img_fn = args['model_name'] + '_iter' + str(num) + '_img' + str(image_id) + '_bs' + str(args['batch_size']) + '-' + str(i + 1) + '.png'
        res[i].save(rslt_img_fn)
        result_md5_list.append(hashlib.md5(Image.open(rslt_img_fn).tobytes()).hexdigest())
    generation_time = end - start
    iter_data = gen_iterate_data(
        iter_idx=num,
        infer_count=nsteps,
        gen_time=generation_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        prompt_idx=image_id,
    )
    iter_data_list.append(iter_data)
    utils.metrics_print.print_metrics(
        num,
        iter_data,
        [],
        generated=rslt_img_fn,
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
    )


def run_image_generation_benchmark(model_path, framework, device, args, num_iters):
    pipe, pretrain_time = FW_UTILS[framework].create_image_gen_model(model_path, device, **args)
    nsteps = int(DEFAULT_INFERENCE_STEPS if args['infer_count'] is None else args['infer_count'])
    iter_data_list = []
    input_text_list = utils.model_utils.get_prompts(args)
    if len(input_text_list) == 0:
        raise RuntimeError('==Failure prompts is empty ==')

    log.info(f'num_iters={num_iters}, num_text_list={len(input_text_list)}')
    # if num_iters == 0, just output warm-up data
    for num in range(num_iters + 1):
        image_id = 0
        for input_text in input_text_list:
            run_image_generation(input_text, 1 if num == 0 else nsteps, num, image_id, pipe, args, iter_data_list)
            image_id += 1

    utils.metrics_print.print_average(iter_data_list)

    return iter_data_list, pretrain_time


def run_image_classification(model_path, framework, device, args, num_iters=10):
    model, input_size = FW_UTILS[framework].create_image_classification_model(model_path, device, **args)

    data = torch.rand(input_size)

    test_time = []
    iter_data_list = []
    for num in range(num_iters or 10):
        start = time.perf_counter()
        model(data)
        end = time.perf_counter()
        generation_time = end - start
        test_time.append(generation_time)

        iter_data = gen_iterate_data(iter_idx=num, in_size=input_size, infer_count=num_iters, gen_time=generation_time)
        iter_data_list.append(iter_data)
    log.info(f'Processed {num_iters} images in {np.sum(test_time)}s')
    log.info(f'Average processing time {np.mean(test_time)} s')
    return iter_data_list


def run_ldm_super_resolution(img, num, nsteps, pipe, args, framework, iter_data_list, image_id):
    set_seed(args['seed'])
    log.info(f'Test {num} input image={img}')
    low_res_img = PIL.Image.open(img).convert('RGB')
    low_res_img = low_res_img.resize((128, 128))
    max_rss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    start = time.perf_counter()
    res = pipe(low_res_img, num_inference_steps=nsteps)
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()
    if num == 0:
        rslt_img_fn = args['model_name'] + '_warmup_' + img.name
    else:
        rslt_img_fn = args['model_name'] + '_iter' + str(num) + '_' + img.name
    log.info(f'Result will be saved to {rslt_img_fn}')
    if framework == 'ov':
        res[0].save(rslt_img_fn)
        md5hash = hashlib.md5(Image.open(rslt_img_fn).tobytes())
    else:
        md5hash = ''

    generation_time = end - start
    iter_data = gen_iterate_data(
        iter_idx=num,
        infer_count=nsteps,
        gen_time=generation_time,
        res_md5=md5hash.hexdigest() if md5hash != '' else '',
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        prompt_idx=image_id,
    )
    iter_data_list.append(iter_data)
    utils.metrics_print.print_metrics(
        num,
        iter_data,
        [],
        generated=rslt_img_fn,
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
    )


def run_ldm_super_resolution_benchmark(model_path, framework, device, args, num_iters):
    pipe, pretrain_time = FW_UTILS[framework].create_ldm_super_resolution_model(model_path, device, **args)
    iter_data_list = []
    input_prompts_list = utils.model_utils.get_prompts(args)
    if len(input_prompts_list) > 0:
        images = []
        for image in input_prompts_list:
            image = os.path.join(os.path.dirname(args['prompt'] if args['prompt'] is not None else args['prompt_file']), image.replace('./', ''))
            images.append(Path(image))
    else:
        images = args['images'] or Path(__file__).parents[0] / 'prompts/test_data/lr_img.png'
        images = Path(images)
        if images.is_dir():
            images = list(images.glob('*'))
        else:
            images = [images]
    log.info(f'Number benchmarking images {len(images)}')
    num_inference_steps = int(DEFAULT_SUPER_RESOLUTION_STEPS if args['infer_count'] is None else args['infer_count'])

    # if num_iters == 0, just output warm-up data
    for num in range(num_iters + 1):
        image_id = 0
        for img in images:
            run_ldm_super_resolution(img, num, 1 if num == 0 else num_inference_steps, pipe, args, framework, iter_data_list, image_id)
            image_id = image_id + 1
    utils.metrics_print.print_average(iter_data_list)

    return iter_data_list, pretrain_time


def num_iters_type(x):
    x = int(x)
    if x < 0:
        raise argparse.ArgumentTypeError('Minimum input value is 0')
    return x


def get_argprser():
    parser = argparse.ArgumentParser('LLM benchmarking tool', add_help=True, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', help='model folder including IR files or Pytorch files', required=TabError)
    parser.add_argument(
        '-id',
        '--model_id',
        default='',
        help='model id of huggingface, if model folder is empty, will try to download model from Hugging Face with this model_id.\n'
        'e.g. the model id of dolly-v2-12b which get from https://huggingface.co/databricks/dolly-v2-12b is databricks/dolly-v2-12b',
    )
    parser.add_argument('-d', '--device', default='cpu', help='inference device')
    parser.add_argument('-r', '--report', help='report csv')
    parser.add_argument('-f', '--framework', default='ov', help='framework')
    parser.add_argument('-p', '--prompt', default=None, help='one prompt')
    parser.add_argument('-pf', '--prompt_file', default=None, help='prompt file in jsonl format')
    parser.add_argument(
        '-ic',
        '--infer_count',
        default=None,
        type=int,
        help='limit the output token size '
        f'(default {DEFAULT_OUTPUT_TOKEN_SIZE}) of text_gen and code_gen models, \n'
        f'or set inference/sampling steps (default {DEFAULT_INFERENCE_STEPS}) of Text2Image models.',
    )
    parser.add_argument(
        '-n',
        '--num_iters',
        default=0,
        type=num_iters_type,
        help='number of benchmarking iterations, '
        'if the value is greater than 0, the average numbers exclude the first(0th) iteration,\n'
        'if the value equals 0 (default), execute the warm-up iteration(0th iteration).',
    )
    parser.add_argument('-i', '--images', default=None, help='test images for vision tasks. Can be directory or path to single image')
    parser.add_argument('-s', '--seed', type=int, default=42, required=False, help='specific random seed to generate fix result. Default 42.')
    parser.add_argument(
        '-lc',
        '--load_config',
        default=None,
        required=False,
        help='path to JSON file to load customized configurations.\n'
        'Example for OpenVINO: {\"INFERENCE_NUM_THREADS\":32,\"PERFORMANCE_HINT\":\"LATENCY\"}.\n'
        'Example for Pytorch: {\"PREC_BF16\":true}. Pytorch currently only supports bf16 settings.\n',
    )
    parser.add_argument(
        '-mc',
        '--memory_consumption',
        default=0,
        required=False,
        type=int,
        help='if the value is 1, output the maximum memory consumption in warm-up iterations. If the value is 2,'
        ' output the maximum memory consumption in all iterations.',
    )
    parser.add_argument('-bs', '--batch_size', type=int, default=1, required=False, help='Batch size value')
    parser.add_argument(
        '--fuse_decoding_strategy',
        action='store_true',
        help='Add decoding postprocessing for next token selection to the model as an extra ops. Original hf_model.generate function will be patched.',
    )
    parser.add_argument(
        '--make_stateful',
        action='store_true',
        help='Replace kv-cache inputs and outputs in the model by internal variables making a stateful model.'
        'Original hf_model.forward function will be patched.',
    )
    parser.add_argument(
        '--save_prepared_model',
        default=None,
        help='Path to .xml file to save IR used for inference with all pre-/post processing included',
    )
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams in the decoding strategy, activates beam_search if greater than 1')
    parser.add_argument(
        '--fuse_cache_reorder',
        action='store_true',
        help='Fuse ops related to cache reordering to the model, applied only when num_beams > 1',
    )
    parser.add_argument(
        '--torch_compile_backend',
        default='openvino',
        required=False,
        help='Enables running the torch.compile() with specified backend: pytorch or openvino (default)',
    )

    return parser.parse_args()


CASE_TO_BENCH = {
    'text_gen': run_text_generation_benchmark,
    'image_gen': run_image_generation_benchmark,
    'image_cls': run_image_classification,
    'code_gen': run_text_generation_benchmark,
    'ldm_super_resolution': run_ldm_super_resolution_benchmark,
}


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = get_argprser()
    model_path, framework, model_args, model_name = utils.model_utils.analyze_args(args)

    # Set the device for running OpenVINO backend for torch.compile()
    if model_args['torch_compile_backend']:
        ov_torch_backend_device = str(args.device)
        os.putenv('OPENVINO_TORCH_BACKEND_DEVICE', ov_torch_backend_device.upper())
        os.system('echo OPENVINO_TORCH_BACKEND_DEVICE=$OPENVINO_TORCH_BACKEND_DEVICE')

    if framework == 'ov':
        log.info(f'model_path={model_path}, openvino runtime version:{get_version()}')
        if model_args['config'].get('PREC_BF16') and model_args['config']['PREC_BF16'] is True:
            log.warning('[Warning] Param bf16/prec_bf16 only work for framework pt. It will be disabled.')
    if args.memory_consumption:
        mem_consumption.start_collect_mem_consumption_thread()
    try:
        iter_data_list, pretrain_time = CASE_TO_BENCH[model_args['use_case']](model_path, framework, args.device, model_args, args.num_iters)
        if args.report is not None:
            model_precision = ''
            if framework == 'ov':
                ir_conversion_frontend = utils.model_utils.get_ir_conversion_frontend(model_name, model_path.parents._parts)
                if ir_conversion_frontend != '':
                    framework = framework + '(' + ir_conversion_frontend + ')'
                model_precision = utils.model_utils.get_model_precision(model_path.parents._parts)
            utils.output_csv.write_result(
                args.report,
                model_name,
                framework,
                args.device,
                model_args['use_case'],
                iter_data_list,
                pretrain_time,
                model_precision,
            )
    except Exception:
        log.error('An exception occurred')
        log.info(traceback.format_exc())
    finally:
        if args.memory_consumption:
            mem_consumption.end_collect_mem_consumption_thread()


if __name__ == '__main__':
    main()
