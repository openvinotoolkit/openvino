#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
This sample demonstrates INT8 post-training quantization (PTQ) of a
vision-transformer (ViT) backbone using NNCF, and benchmarks FP32 vs INT8
latency and top-1 accuracy on Intel iGPU.

Resolves: https://github.com/openvinotoolkit/openvino/issues/35023
"""

import logging as log
import sys
import time
import numpy as np
import openvino as ov
import nncf
import torch
from transformers import ViTForImageClassification
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

log.basicConfig(format='[ %(levelname)s ] %(message)s',
                level=log.INFO, stream=sys.stdout)


def load_model():
    log.info('Loading ViT-B/16 from Hugging Face...')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224')
    model.eval()
    return model


def convert_to_openvino(model):
    log.info('Converting to OpenVINO IR...')
    dummy_input = torch.randn(1, 3, 224, 224)
    ov_model = ov.convert_model(model, example_input=dummy_input)
    ov.save_model(ov_model, 'vit_fp32.xml')
    log.info('Saved vit_fp32.xml')
    return ov_model


def build_calibration_loader(imagenet_path, num_samples=300):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageNet(imagenet_path, split='val',
                                transform=transform)
    subset = torch.utils.data.Subset(dataset, range(num_samples))
    return DataLoader(subset, batch_size=32, shuffle=False)


def quantize_model(ov_model, calibration_loader):
    log.info('Applying NNCF INT8 PTQ...')

    def transform_fn(data_item):
        return {'pixel_values': data_item[0].numpy()}

    nncf_dataset = nncf.Dataset(calibration_loader, transform_fn)
    quantized_model = nncf.quantize(ov_model, nncf_dataset)
    ov.save_model(quantized_model, 'vit_int8.xml')
    log.info('Saved vit_int8.xml')
    return quantized_model


def benchmark_latency(compiled_model, device, num_runs=100):
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    # Warm up
    for _ in range(10):
        compiled_model({'pixel_values': dummy})
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        compiled_model({'pixel_values': dummy})
    elapsed = (time.perf_counter() - start) / num_runs * 1000
    log.info(f'[{device}] Average latency over {num_runs} runs: '
             f'{elapsed:.2f} ms')
    return elapsed


def evaluate_accuracy(compiled_model, dataloader, label=''):
    correct = 0
    total = 0
    for images, labels in dataloader:
        output = compiled_model({'pixel_values': images.numpy()})[0]
        preds = output.argmax(axis=1)
        correct += (preds == labels.numpy()).sum()
        total += len(labels)
    acc = correct / total
    log.info(f'[{label}] Top-1 Accuracy: {acc:.4f}')
    return acc


def main():
    if len(sys.argv) < 2:
        log.info(f'Usage: {sys.argv[0]} <path_to_imagenet_val> '
                 f'[device]')
        log.info('  device: CPU (default) or GPU for Intel iGPU')
        return 1

    imagenet_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else 'CPU'

    #Step 1. Load and convert model 
    model = load_model()
    ov_model = convert_to_openvino(model)

    #Step 2. Build calibration data
    calibration_loader = build_calibration_loader(imagenet_path)

    #Step 3. Quantize
    quantized_model = quantize_model(ov_model, calibration_loader)

    #Step 4. Compile models for target device
    core = ov.Core()
    log.info(f'Available devices: {core.available_devices}')

    log.info(f'Compiling FP32 model on {device}...')
    fp32_compiled = core.compile_model('vit_fp32.xml', device)

    log.info(f'Compiling INT8 model on {device}...')
    int8_compiled = core.compile_model('vit_int8.xml', device)

    #Step 5. Benchmark latency
    fp32_latency = benchmark_latency(fp32_compiled, f'FP32/{device}')
    int8_latency = benchmark_latency(int8_compiled, f'INT8/{device}')
    speedup = fp32_latency / int8_latency
    log.info(f'Speedup (FP32 -> INT8): {speedup:.2f}x')

    #Step 6. Evaluate accuracy
    eval_loader = build_calibration_loader(imagenet_path, num_samples=500)
    fp32_acc = evaluate_accuracy(fp32_compiled, eval_loader, 'FP32')
    int8_acc = evaluate_accuracy(int8_compiled, eval_loader, 'INT8')
    acc_drop = fp32_acc - int8_acc
    log.info(f'Accuracy drop (FP32 -> INT8): {acc_drop:.4f}')

    #Step 7. Print results table
    log.info('\n')
    log.info('=' * 50)
    log.info(f'  Results on {device}')
    log.info('=' * 50)
    log.info(f'  {"Precision":<12} {"Latency (ms)":<16} {"Top-1 Acc":<12}')
    log.info(f'  {"-"*40}')
    log.info(f'  {"FP32":<12} {fp32_latency:<16.2f} {fp32_acc:<12.4f}')
    log.info(f'  {"INT8":<12} {int8_latency:<16.2f} {int8_acc:<12.4f}')
    log.info('=' * 50)

    return 0


if __name__ == '__main__':
    sys.exit(main())