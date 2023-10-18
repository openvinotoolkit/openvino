# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import sys
import gc
import time
import logging as log
from argparse import ArgumentParser
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Tuple
import torch
from diffusers import StableDiffusionPipeline, LDMSuperResolutionPipeline
from nncf import compress_weights
from openvino import Type, PartialShape, save_model, convert_model
from openvino.runtime import Core
from optimum.exporters import TasksManager
from optimum.exporters.tasks import make_backend_config_constructor_for_task
from optimum.exporters.onnx.config import TextDecoderOnnxConfig
from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
from optimum.exporters.onnx import get_encoder_decoder_models_for_export
from optimum.exporters.openvino import export_models
from optimum.intel.openvino import (
    OVModelForCausalLM,
    OVModelForSeq2SeqLM,
    OVStableDiffusionPipeline,
    OVQuantizer,
    OV_XML_FILE_NAME,
    OV_DECODER_NAME,
    OV_DECODER_WITH_PAST_NAME,
    OV_ENCODER_NAME,
)
from optimum.exporters.onnx import __main__ as optimum_main
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel


class BackendType(Enum):
    PYTORCH = 'pytorch'
    OPENVINO = 'openvino'


def save_tokenizer(tokenizer, out_dir):
    try:
        tokenizer.save_pretrained(out_dir)
    except Exception:
        log.error('tokenizer loading failed')


def compress_ov_model_weights_helper(ov_model, tok, config, out_path, fp16=False):
    compressed_ov_model = compress_weights(ov_model)
    save_ov_model_helper(compressed_ov_model, out_path, fp16=fp16, tok=tok, config=config)


def save_ov_model_helper(ov_model, out_path, model_name='openvino_model', fp16=False, tok=None, config=None):
    save_model(ov_model, Path(out_path) / f'{model_name}.xml', compress_to_fp16=fp16)
    if tok is not None:
        save_tokenizer(tok, out_path)
    if config is not None:
        config.save_pretrained(out_path)


def convert_causal_lm(args):
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    pt_compress_weights = args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends

    start = time.perf_counter()
    if args.save_orig or pt_compress_weights:
        pt_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
        )
        if args.save_orig:
            pt_out_dir = Path(args.output_dir) / 'pytorch'
            pt_model.save_pretrained(pt_out_dir)
            save_tokenizer(tok, pt_out_dir)
        if pt_compress_weights:
            feature = 'text-generation'
            quantizer = OVQuantizer.from_pretrained(pt_model, task=feature)
            pt_out_dir = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
            quantizer.quantize(save_directory=pt_out_dir, weights_only=True)
            save_tokenizer(tok, pt_out_dir)
        del pt_model
        gc.collect()

    model = OVModelForCausalLM.from_pretrained(
        args.model_id,
        export=True,
        compile=False,
        trust_remote_code=True,
        config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
    )
    end = time.perf_counter()

    log.info(f'Conversion total time {end - start}s')
    if args.precision == 'FP16':
        model.half()
    ov_out_dir = Path(args.output_dir) / 'pytorch/dldt' / args.precision
    save_tokenizer(tok, ov_out_dir)

    start1 = time.perf_counter()
    model.save_pretrained(ov_out_dir)
    end1 = time.perf_counter()
    log.info(f'Serialization total time {end1 - start1}s')

    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends:
        ov_int8_dir = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'OV_{args.precision}-INT8'
        model.model = compress_weights(model.model)
        model.save_pretrained(ov_int8_dir)
        save_tokenizer(tok, ov_int8_dir)

    del model
    gc.collect()


def convert_seq2seq(args):
    tokenizer_id = args.model_id if 'blenderbot-9B' not in args.model_id else 'facebook/blenderbot-3B'
    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    pt_compress_weights = args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends
    start = time.perf_counter()
    if args.save_orig or pt_compress_weights:
        pt_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
        )
        if args.save_orig:
            pt_out_dir = Path(args.output_dir) / 'pytorch'
            pt_model.save_pretrained(pt_out_dir)
            save_tokenizer(tok, pt_out_dir)
        if pt_compress_weights:
            compressed_pt_model = compress_weights(pt_model)
            onnx_config_constructor = TasksManager.get_exporter_config_constructor(model=pt_model, exporter='onnx', task='text2text-generation')
            onnx_config = onnx_config_constructor(pt_model.config, use_past=True)
            models_and_onnx_configs = get_encoder_decoder_models_for_export(compressed_pt_model, onnx_config)
            encoder_file_name = Path('encoder') / OV_ENCODER_NAME
            decoder_file_name = Path('decoder') / OV_DECODER_NAME
            decoder_with_past_file_name = Path('decoder_with_past') / OV_DECODER_WITH_PAST_NAME

            output_names = [encoder_file_name, decoder_file_name, decoder_with_past_file_name]
            save_dir_path = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
            try:
                export_models(
                    models_and_onnx_configs=models_and_onnx_configs,
                    opset=onnx_config.DEFAULT_ONNX_OPSET,
                    output_dir=save_dir_path,
                    output_names=output_names,
                )
                save_tokenizer(tok, save_dir_path)
            except Exception as ex:
                log.warning(f'PT weights compression failed with {ex}, please use OpenVINO backend instead')

        del pt_model
        gc.collect()

    model = OVModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        export=True,
        compile=False,
        trust_remote_code=True,
        config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
    )
    end = time.perf_counter()
    log.info(f'Conversion total time {end - start}s')

    start1 = time.perf_counter()
    ov_out_dir = Path(args.output_dir) / 'pytorch/dldt' / args.precision
    model.save_pretrained(ov_out_dir)
    end1 = time.perf_counter()
    log.info(f'Serialization total time {end1 - start1}s')

    save_tokenizer(tok, ov_out_dir)

    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends:
        ov_int8_dir = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'OV_{args.precision}-INT8'
        model.encoder.model = compress_weights(model.encoder.model)
        model.decoder.model = compress_weights(model.decoder.model)
        if model.decoder_with_past:
            model.decoder_with_past.model = compress_weights(model.decoder_with_past.model)
        model.save_pretrained(ov_int8_dir)
        save_tokenizer(tok, ov_int8_dir)

    del model
    gc.collect()


def convert_sd(args):
    start = time.perf_counter()
    pt_compress_weights = args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends
    if args.save_orig or pt_compress_weights:
        pt_model = StableDiffusionPipeline.from_pretrained(args.model_id)
        if args.save_orig:
            pt_model.save_pretrained(Path(args.output_dir) / 'pytorch')
        if pt_compress_weights:
            wc_text_encoder = compress_weights(pt_model.text_encoder)
            wc_unet = compress_weights(pt_model.unet)
            wc_vae = compress_weights(pt_model.vae)
            pt_model.text_encoder = wc_text_encoder
            pt_model.unet = wc_unet
            pt_model.vae = wc_vae
            _, models_and_onnx_configs = optimum_main._get_submodels_and_onnx_configs(
                model=pt_model,
                task='stable-diffusion',
                monolith=False,
                custom_onnx_configs={},
                custom_architecture=False,
                _variant='default',
            )
            output = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
            for model_name in models_and_onnx_configs:
                subcomponent = models_and_onnx_configs[model_name][0]
                if hasattr(subcomponent, 'save_config'):
                    subcomponent.save_config(output / model_name)
                elif hasattr(subcomponent, 'config') and hasattr(subcomponent.config, 'save_pretrained'):
                    subcomponent.config.save_pretrained(output / model_name)

            files_subpaths = [Path(name_dir) / OV_XML_FILE_NAME for name_dir in models_and_onnx_configs]

            # Saving the additional components needed to perform inference.
            pt_model.scheduler.save_pretrained(output.joinpath('scheduler'))

            feature_extractor = getattr(pt_model, 'feature_extractor', None)
            if feature_extractor is not None:
                feature_extractor.save_pretrained(output.joinpath('feature_extractor'))

            tokenizer = getattr(pt_model, 'tokenizer', None)
            if tokenizer is not None:
                tokenizer.save_pretrained(output.joinpath('tokenizer'))

            tokenizer_2 = getattr(pt_model, 'tokenizer_2', None)
            if tokenizer_2 is not None:
                tokenizer_2.save_pretrained(output.joinpath('tokenizer_2'))

            pt_model.save_config(output)

            export_models(
                models_and_onnx_configs=models_and_onnx_configs,
                output_dir=output,
                output_names=files_subpaths,
            )

        del pt_model
        gc.collect()

    model = OVStableDiffusionPipeline.from_pretrained(args.model_id, export=True, compile=False)
    end = time.perf_counter()
    log.info(f'Conversion total time {end - start}s')

    if args.precision == 'FP16':
        model.half()
    start1 = time.perf_counter()
    model.save_pretrained(Path(args.output_dir) / 'pytorch/dldt' / args.precision)
    end1 = time.perf_counter()
    log.info(f'Serialization total time {end1 - start1}s')

    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends:
        ov_int8_dir = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'OV_{args.precision}-INT8'
        model.text_encoder.model = compress_weights(model.text_encoder.model)
        model.unet.model = compress_weights(model.unet.model)
        model.vae_decoder.model = compress_weights(model.vae_decoder.model)
        model.save_pretrained(ov_int8_dir)

        # Saving the additional components needed to perform inference.
        model.scheduler.save_pretrained(ov_int8_dir.joinpath('scheduler'))

        feature_extractor = getattr(model, 'feature_extractor', None)
        if feature_extractor is not None:
            feature_extractor.save_pretrained(ov_int8_dir.joinpath('feature_extractor'))

        tokenizer = getattr(model, 'tokenizer', None)
        if tokenizer is not None:
            tokenizer.save_pretrained(ov_int8_dir.joinpath('tokenizer'))

        tokenizer_2 = getattr(model, 'tokenizer_2', None)
        if tokenizer_2 is not None:
            tokenizer_2.save_pretrained(ov_int8_dir.joinpath('tokenizer_2'))

        model.save_config(ov_int8_dir)

    del model
    gc.collect()


def convert_ldm_super_res(args):
    pipeline = LDMSuperResolutionPipeline.from_pretrained(args.model_id)
    if args.save_orig:
        pipeline.save_pretrained(Path(args.output_dir) / 'pytorch')
    unet_example_input = [torch.zeros((1, 6, 128, 128)), torch.tensor(1, dtype=torch.int32)]

    class Decoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, latents):
            return self.model.decode(latents)

    decoder = Decoder(pipeline.vqvae)

    pt_compress_weights = args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends
    compress_to_fp16 = args.precision == 'FP16'
    if pt_compress_weights:
        compressed_unet = compress_weights(pipeline.unet)
        ov_compressed_unet = convert_model(compressed_unet, example_input=unet_example_input)
        ov_compressed_unet.inputs[1].get_node().set_element_type(Type.i32)
        ov_compressed_unet.inputs[1].get_node().set_partial_shape(PartialShape([]))
        ov_compressed_unet.validate_nodes_and_infer_types()
        pt_out_dir = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
        save_model(ov_compressed_unet, pt_out_dir / 'unet.xml', compress_to_fp16=compress_to_fp16)
        pipeline.scheduler.save_config(pt_out_dir)
        # Couldn't compress decoder weights (RuntimeError: cdist only supports floating-point dtypes, X2 got: Byte)
        ov_decoder = convert_model(decoder, example_input=torch.zeros((1, 3, 128, 128)))
        save_model(ov_decoder, pt_out_dir / 'vqvae.xml', compress_to_fp16=compress_to_fp16)

    # convert model to OpenVINO IR
    ov_unet = convert_model(pipeline.unet, example_input=unet_example_input)
    ov_unet.inputs[1].get_node().set_element_type(Type.i32)
    ov_unet.inputs[1].get_node().set_partial_shape(PartialShape([]))
    ov_unet.validate_nodes_and_infer_types()
    save_dir = Path(args.output_dir) / 'pytorch/dldt' / args.precision
    save_model(ov_unet, save_dir / 'unet.xml', compress_to_fp16=compress_to_fp16)
    ov_decoder = convert_model(decoder, example_input=torch.zeros((1, 3, 128, 128)))
    save_model(ov_decoder, save_dir / 'vqvae.xml', compress_to_fp16=compress_to_fp16)
    pipeline.scheduler.save_config(save_dir)

    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends:
        ov_int8_dir = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'OV_{args.precision}-INT8'
        compressed_ov_unet = compress_weights(ov_unet)
        save_model(compressed_ov_unet, ov_int8_dir / 'unet.xml', compress_to_fp16=compress_to_fp16)
        compressed_ov_decoder = compress_weights(ov_decoder)
        save_model(compressed_ov_decoder, ov_int8_dir / 'vqvae.xml', compress_to_fp16=compress_to_fp16)
        pipeline.scheduler.save_config(ov_int8_dir)


def convert_mpt(args):
    def convert_to_ov(pt_model, tok, out_path, compress_to_fp16=False):
        pt_model.config.use_cache = True
        outs = pt_model(input_ids=torch.ones((1, 10), dtype=torch.long), attention_mask=torch.ones((1, 10), dtype=torch.long))
        old = outs.past_key_values[0][0].ndim == 3
        inputs = ['input_ids']
        outputs = ['logits']

        dynamic_shapes = {'input_ids': {1: 'seq_len'}, 'attention_mask': {1: 'seq_len'}}

        for idx in range(len(outs.past_key_values)):
            inputs.extend([f'past_key_values.{idx}.key', f'past_key_values.{idx}.value'])
            dynamic_shapes[inputs[-1]] = {2: 'past_sequence + sequence'}
            dynamic_shapes[inputs[-2]] = {3 if not old else 2: 'past_sequence + sequence'}
            outputs.extend([f'present.{idx}.key', f'present.{idx}.value'])

        inputs.append('attention_mask')
        dummy_inputs = {
            'input_ids': torch.ones((1, 2), dtype=torch.long),
            'past_key_values': outs.past_key_values,
            'attention_mask': torch.ones((1, 12), dtype=torch.long),
        }
        pt_model.config.torchscript = True
        orig_forward = pt_model.forward

        @wraps(orig_forward)
        def ts_patched_forward(input_ids: torch.Tensor, past_key_values: Tuple[Tuple[torch.Tensor]], attention_mask: torch.Tensor):
            pkv_list = list(past_key_values)
            outs = orig_forward(input_ids=input_ids, past_key_values=pkv_list, attention_mask=attention_mask)
            return (outs.logits, tuple(outs.past_key_values))

        pt_model.forward = ts_patched_forward
        ov_model = convert_model(pt_model, example_input=dummy_inputs)
        pt_model.forward = orig_forward

        for inp_name, m_input, input_data in zip(inputs, ov_model.inputs, flattenize_inputs(dummy_inputs.values())):
            input_node = m_input.get_node()
            if input_node.element_type == Type.dynamic:
                m_input.get_node().set_element_type(Type.f32)
            shape = list(input_data.shape)
            if inp_name in dynamic_shapes:
                for k in dynamic_shapes[inp_name]:
                    shape[k] = -1
            input_node.set_partial_shape(PartialShape(shape))
            m_input.get_tensor().set_names({inp_name})

        for out, out_name in zip(ov_model.outputs, outputs):
            out.get_tensor().set_names({out_name})

        save_ov_model_helper(ov_model, out_path, fp16=compress_to_fp16, tok=tok, config=pt_model.config)

    pt_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
    )
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    pt_model.config.use_cache = True
    pt_model.eval()

    if args.save_orig:
        pt_out_dir = Path(args.output_dir) / 'pytorch'
        pt_model.save_pretrained(pt_out_dir)
        save_tokenizer(tok, pt_out_dir)

    ov_dir = Path(args.output_dir) / 'pytorch/dldt' / args.precision
    compress_to_fp16 = args.precision == 'FP16'

    convert_to_ov(pt_model, tok, ov_dir, compress_to_fp16)
    if args.compress_weights:
        if BackendType.PYTORCH.value in args.compress_weights_backends:
            compressed_pt_model = compress_weights(pt_model)
            pt_path = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
            convert_to_ov(compressed_pt_model, tok, pt_path, compress_to_fp16)
        if BackendType.OPENVINO.value in args.compress_weights_backends:
            ov_model = Core().read_model(ov_dir / 'openvino_model.xml')
            ov_compressed_path = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'OV_{args.precision}-INT8'
            compress_ov_model_weights_helper(ov_model, tok, pt_model.config, ov_compressed_path, compress_to_fp16)


def convert_stablelm(args):
    def convert_to_ov(pt_model, tok, out_path, compress_to_fp16=False):
        pt_model.config.use_cache = True
        outs = pt_model(input_ids=torch.ones((1, 10), dtype=torch.long), attention_mask=torch.ones((1, 10), dtype=torch.long))
        inputs = ['input_ids', 'attention_mask']
        outputs = ['logits']

        dynamic_shapes = {'input_ids': {1: 'seq_len'}, 'attention_mask': {1: 'seq_len'}}

        for idx in range(len(outs.past_key_values)):
            inputs.extend([f'past_key_values.{idx}.key', f'past_key_values.{idx}.value'])
            dynamic_shapes[inputs[-1]] = {2: 'past_sequence + sequence'}
            dynamic_shapes[inputs[-2]] = {2: 'past_sequence + sequence'}
            outputs.extend([f'present.{idx}.key', f'present.{idx}.value'])
        dummy_inputs = {
            'input_ids': torch.ones((1, 2), dtype=torch.long),
            'attention_mask': torch.ones((1, 12), dtype=torch.long),
            'past_key_values': outs.past_key_values,
        }
        pt_model.config.torchscript = True
        ov_model = convert_model(pt_model, example_input=dummy_inputs)

        for inp_name, m_input, input_data in zip(inputs, ov_model.inputs, flattenize_inputs(dummy_inputs.values())):
            input_node = m_input.get_node()
            if input_node.element_type == Type.dynamic:
                m_input.get_node().set_element_type(Type.f32)
            shape = list(input_data.shape)
            if inp_name in dynamic_shapes:
                for k in dynamic_shapes[inp_name]:
                    shape[k] = -1
            input_node.set_partial_shape(PartialShape(shape))
            m_input.get_tensor().set_names({inp_name})

        for out, out_name in zip(ov_model.outputs, outputs):
            out.get_tensor().set_names({out_name})

        save_ov_model_helper(ov_model, out_path, fp16=compress_to_fp16, tok=tok, config=pt_model.config)

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    if not config.model_type.startswith('stablelm'):
        return convert_causal_lm(args)
    pt_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
    )
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    pt_model.config.use_cache = True
    pt_model.eval()

    if args.save_orig:
        pt_out_dir = Path(args.output_dir) / 'pytorch'
        pt_model.save_pretrained(pt_out_dir)
        save_tokenizer(tok, pt_out_dir)

    ov_dir = Path(args.output_dir) / 'pytorch/dldt' / args.precision
    compress_to_fp16 = args.precision == 'FP16'

    convert_to_ov(pt_model, tok, ov_dir, compress_to_fp16)
    if args.compress_weights:
        if BackendType.PYTORCH.value in args.compress_weights_backends:
            compressed_pt_model = compress_weights(pt_model)
            pt_path = Path(args.output_dir) / 'pytorch/dldt/PT_compressed_weights'
            convert_to_ov(compressed_pt_model, tok, pt_path, compress_to_fp16)
        if BackendType.OPENVINO.value in args.compress_weights_backends:
            ov_model = Core().read_model(ov_dir / 'openvino_model.xml')
            ov_compressed_path = Path(args.output_dir) / 'pytorch/dldt/INT8_compressed_weights'
            compress_ov_model_weights_helper(ov_model, tok, pt_model.config, ov_compressed_path, compress_to_fp16)


def convert_chatglm2(args):
    def convert_to_ov(pt_model, tok, out_path, compress_to_fp16=False):
        pt_model.config.torchscript = True
        pt_model.config.use_cache = True
        outs = pt_model(input_ids=torch.ones((1, 10), dtype=torch.long), position_ids=torch.arange(0, 10, dtype=torch.long))
        inputs = ['input_ids', 'position_ids']
        outputs = ['logits']
        dynamic_shapes = {'input_ids': {1: 'seq_len'}, 'position_ids': {1: 'seq_len'}}
        for idx in range(len(outs[1])):
            inputs.extend([f'past_key_values.{idx}.key', f'past_key_values.{idx}.value'])
            dynamic_shapes[inputs[-1]] = {0: 'past_sequence + 1'}
            dynamic_shapes[inputs[-2]] = {0: 'past_sequence + 1'}
            outputs.extend([f'present.{idx}.key', f'present.{idx}.value'])
        dummy_inputs = {
            'input_ids': torch.ones((1, 1), dtype=torch.long),
            'position_ids': torch.tensor([[10]], dtype=torch.long),
            'past_key_values': outs[1],
        }
        ov_model = convert_model(pt_model, example_input=dummy_inputs)
        for inp_name, m_input, input_data in zip(inputs, ov_model.inputs, flattenize_inputs(dummy_inputs.values())):
            input_node = m_input.get_node()
            if input_node.element_type == Type.dynamic:
                m_input.get_node().set_element_type(Type.f32)
            shape = list(input_data.shape)
            if inp_name in dynamic_shapes:
                for k in dynamic_shapes[inp_name]:
                    shape[k] = -1
            input_node.set_partial_shape(PartialShape(shape))
            m_input.get_tensor().set_names({inp_name})

        for out, out_name in zip(ov_model.outputs, outputs):
            out.get_tensor().set_names({out_name})
        ov_model.validate_nodes_and_infer_types()
        save_ov_model_helper(ov_model, out_path, fp16=compress_to_fp16, tok=tok, config=pt_model.config)

    pt_model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
    )
    pt_model.config.use_cache = True
    pt_model.to(torch.float32)
    pt_model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if args.save_orig:
        pt_out_dir = Path(args.output_dir) / 'pytorch'
        pt_model.save_pretrained(pt_out_dir)
        save_tokenizer(tok, pt_out_dir)

    compress_to_fp16 = args.precision == 'FP16'
    ov_out_path = Path(args.output_dir) / 'pytorch/dldt' / args.precision
    convert_to_ov(pt_model, tok, ov_out_path, compress_to_fp16=compress_to_fp16)

    pt_compress_weights = args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends
    if pt_compress_weights:
        compressed_pt_model = compress_weights(pt_model)
        pt_out_path = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
        convert_to_ov(compressed_pt_model, tok, pt_out_path, compress_to_fp16=compress_to_fp16)

    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends:
        ov_model_path = ov_out_path / 'openvino_model.xml'
        ov_model = Core().read_model(ov_model_path)
        ov_compressed_path = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'OV_{args.precision}-INT8'
        compress_ov_model_weights_helper(ov_model, tok, pt_model.config, ov_compressed_path, compress_to_fp16)


def convert_chatglm(args):
    def convert_to_ov(pt_model, tok, out_path, compress_to_fp16=False):
        pt_model.config.torchscript = True
        last_token = torch.tensor([[130328]])
        past = torch.zeros(28, 2, 5, 1, 32, 128)
        position_ids = torch.tensor([[[2], [4]]])
        dummy_input = {
            'input_ids': last_token,
            'past_key_values': past,
            'position_ids': position_ids,
        }
        ov_model = convert_model(pt_model, example_input=dummy_input)
        ov_model.outputs[0].get_tensor().set_names({'logits'})
        for i in range(1, len(ov_model.outputs), 2):
            idx = (i - 1) // 2
            ov_model.outputs[i].get_tensor().set_names({f'present.{int(idx)}.key'})
            ov_model.outputs[i + 1].get_tensor().set_names({f'present.{int(idx)}.value'})
        save_ov_model_helper(ov_model, out_path, fp16=compress_to_fp16, tok=tok, config=pt_model.config)

    pt_model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
    )
    pt_model.config.use_cache = True
    pt_model.to(torch.float32)
    pt_model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if args.save_orig:
        pt_out_dir = Path(args.output_dir) / 'pytorch'
        pt_model.save_pretrained(pt_out_dir)
        save_tokenizer(tok, pt_out_dir)

    compress_to_fp16 = args.precision == 'FP16'
    ov_out_path = Path(args.output_dir) / 'pytorch/dldt' / args.precision
    convert_to_ov(pt_model, tok, ov_out_path, compress_to_fp16=compress_to_fp16)

    pt_compress_weights = args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends
    if pt_compress_weights:
        compressed_pt_model = compress_weights(pt_model)
        pt_out_path = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
        convert_to_ov(compressed_pt_model, tok, pt_out_path)

    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends:
        ov_model_path = ov_out_path / 'openvino_model.xml'
        ov_model = Core().read_model(ov_model_path)
        ov_compressed_path = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'OV_{args.precision}-INT8'
        compress_ov_model_weights_helper(ov_model, tok, pt_model.config, ov_compressed_path, compress_to_fp16)


def flattenize_inputs(inputs):
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


def convert_falcon(args):
    def convert_to_ov(pt_model, tok, out_path, compress_to_fp16=False):
        outs = pt_model(input_ids=torch.ones((1, 10), dtype=torch.long))
        inputs = ['input_ids']
        outputs = ['logits']

        dynamic_shapes = {'input_ids': {1: 'seq_len'}}

        for idx in range(len(outs.past_key_values)):
            inputs.extend([f'past_key_values.{idx}.key', f'past_key_values.{idx}.value'])
            dynamic_shapes[inputs[-1]] = {1: 'past_sequence + sequence'}
            dynamic_shapes[inputs[-2]] = {1: 'past_sequence + sequence'}
            outputs.extend([f'present.{idx}.key', f'present.{idx}.value'])

        dummy_inputs = {'input_ids': torch.ones((1, 2), dtype=torch.long), 'past_key_values': outs.past_key_values}
        flatten_inputs = flattenize_inputs(dummy_inputs.values())
        pt_model.config.torchscript = True
        ov_model = convert_model(pt_model, example_input=dummy_inputs)
        for port, input_data, input_name in zip(ov_model.inputs[1:], flatten_inputs[1:], inputs[1:]):
            port.get_node().set_element_type(Type.f32)
            shape = list(input_data.shape)
            shape[2] = -1
            port.get_node().set_partial_shape(PartialShape(shape))
            port.get_tensor().set_names({input_name})
        for idx, out_name in enumerate(outputs):
            ov_model.outputs[idx].get_tensor().set_names({out_name})
        ov_model.validate_nodes_and_infer_types()
        save_ov_model_helper(ov_model, out_path, fp16=compress_to_fp16, tok=tok, config=pt_model.config)

    pt_model = AutoModelForCausalLM.from_pretrained(args.model_id, config=AutoConfig.from_pretrained(args.model_id))
    pt_model.config.use_cache = True
    pt_model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if args.save_orig:
        pt_out_dir = Path(args.output_dir) / 'pytorch'
        pt_model.save_pretrained(pt_out_dir)
        save_tokenizer(tok, pt_out_dir)

    compress_to_fp16 = args.precision == 'FP16'

    ov_out_path = Path(args.output_dir) / 'pytorch/dldt' / args.precision
    convert_to_ov(pt_model, tok, ov_out_path, compress_to_fp16)

    if args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends:
        pt_compressed_model = compress_weights(pt_model)
        pt_comp_path = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
        convert_to_ov(pt_compressed_model, tok, pt_comp_path, compress_to_fp16)

    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends:
        ov_model = Core().read_model(ov_out_path / 'openvino_model.xml')
        ov_compressed_path = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'OV_{args.precision}-INT8'
        compress_ov_model_weights_helper(ov_model, tok, pt_model.config, ov_compressed_path, compress_to_fp16)


def convert_jais(args):
    normalized_config = NormalizedTextConfig.with_args(num_layers='n_layer', num_attention_heads='n_head', hidden_size='n_embd')

    class JaisOpenVINOConfig(TextDecoderOnnxConfig):
        DEFAULT_ONNX_OPSET = 13
        NORMALIZED_CONFIG_CLASS = normalized_config

    TasksManager._SUPPORTED_MODEL_TYPE['jais'] = {
        'onnx': {
            'text-generation': make_backend_config_constructor_for_task(JaisOpenVINOConfig, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(JaisOpenVINOConfig, 'text-generation-with-past'),
        },
    }
    NormalizedConfigManager._conf['jais'] = normalized_config
    return convert_causal_lm(args)


converters = {
    'decoder': convert_causal_lm,
    'blenderbot': convert_seq2seq,
    't5': convert_seq2seq,
    'stable-diffusion': convert_sd,
    'ldm': convert_ldm_super_res,
    'mpt': convert_mpt,
    'replit': convert_mpt,
    'chatglm2': convert_chatglm2,
    'chatglm': convert_chatglm,
    'falcon': convert_falcon,
    'stablelm': convert_stablelm,
    'jais': convert_jais,
}


def get_convert_model_type(model_id):
    default = 'decoder'
    for key in converters:
        if key in model_id:
            return key

    return default


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    parser = ArgumentParser()
    parser.add_argument('--model_id', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--save_orig', action='store_true')
    parser.add_argument('--precision', choices=['FP32', 'FP16'], default='FP32')

    compression_group = parser.add_argument_group('Weights compression parameters')
    compression_group.add_argument('--compress_weights', action='store_true')
    compression_group.add_argument(
        '--compress_weights_backends',
        help='Backend names used to compress the input model weights separated by space.',
        choices=[BackendType.PYTORCH.value, BackendType.OPENVINO.value],
        default=BackendType.OPENVINO.value,
        type=str.lower,
        nargs='+',
    )

    args = parser.parse_args()
    model_type = get_convert_model_type(args.model_id)
    converter = converters[model_type]
    converter(args)


main()
