#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import openvino as ov
import datasets
from transformers import AutoTokenizer
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    log.info('OpenVINO:')
    log.info(f"{'Build ':.<39} {ov.__version__}")
    model_name = 'bert-base-uncased'
    # Download the model
    transformers_model = FeaturesManager.get_model_from_feature('default', model_name)
    _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(transformers_model, feature='default')
    onnx_config = model_onnx_config(transformers_model.config)
    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    core = ov.Core()

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / f'{model_name}.onnx'
        # Export .onnx
        export(tokenizer, transformers_model, onnx_config, onnx_config.default_onnx_opset, onnx_path)
        # Read .onnx with OpenVINO
        model = core.read_model(onnx_path)

    # Enforce dynamic input shape
    try:
        model.reshape({model_input.any_name: ov.PartialShape([1, '?']) for model_input in model.inputs})
    except RuntimeError:
        log.error("Can't set dynamic shape")
        raise
    # Optimize for throughput. Best throughput can be reached by
    # running multiple openvino.runtime.InferRequest instances asyncronously
    tput = {'PERFORMANCE_HINT': 'THROUGHPUT'}
    # Pick a device by replacing CPU, for example MULTI:CPU(4),GPU(8).
    # It is possible to set CUMULATIVE_THROUGHPUT as PERFORMANCE_HINT for AUTO device
    compiled_model = core.compile_model(model, 'CPU', tput)
    # AsyncInferQueue creates optimal number of InferRequest instances
    ireqs = ov.AsyncInferQueue(compiled_model)

    sst2 = datasets.load_dataset('glue', 'sst2')
    sst2_sentences = sst2['validation']['sentence']
    # Warm up
    encoded_warm_up = dict(tokenizer('Warm up sentence is here.', return_tensors='np'))
    for _ in range(len(ireqs)):
        ireqs.start_async(encoded_warm_up)
    ireqs.wait_all()
    # Benchmark
    sum_seq_len = 0
    start = perf_counter()
    for sentence in sst2_sentences:
        encoded = dict(tokenizer(sentence, return_tensors='np'))
        sum_seq_len += next(iter(encoded.values())).size  # get sequence length to compute average length
        ireqs.start_async(encoded)
    ireqs.wait_all()
    end = perf_counter()
    duration = end - start
    log.info(f'Average sequence length: {sum_seq_len / len(sst2_sentences):.2f}')
    log.info(f'Average processing time: {duration / len(sst2_sentences) * 1e3:.2f} ms')
    log.info(f'Duration:                {duration:.2f} seconds')


if __name__ == '__main__':
    main()
