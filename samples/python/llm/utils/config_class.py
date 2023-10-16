# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, T5ForConditionalGeneration, BlenderbotForConditionalGeneration, AutoModel
from diffusers.pipelines import DiffusionPipeline, LDMSuperResolutionPipeline
from optimum.intel.openvino import OVModelForCausalLM, OVModelForSeq2SeqLM, OVStableDiffusionPipeline
from utils.ov_model_classes import OVMPTModel, OVFalconModel, OVLDMSuperResolutionPipeline, OVChatGLMModel, OVChatGLM2Model

TOKENIZE_CLASSES_MAPPING = {
    'decoder': AutoTokenizer,
    'mpt': AutoTokenizer,
    't5': AutoTokenizer,
    'blenderbot': AutoTokenizer,
    'falcon': AutoTokenizer,
}

OV_MODEL_CLASSES_MAPPING = {
    'decoder': OVModelForCausalLM,
    't5': OVModelForSeq2SeqLM,
    'blenderbot': OVModelForSeq2SeqLM,
    'mpt': OVMPTModel,
    'falcon': OVFalconModel,
    'stable_diffusion': OVStableDiffusionPipeline,
    'replit': OVMPTModel,
    'codet5': OVModelForSeq2SeqLM,
    'codegen2': OVModelForCausalLM,
    'ldm_super_resolution': OVLDMSuperResolutionPipeline,
    'chatglm2': OVChatGLM2Model,
    'chatglm': OVChatGLMModel,
}

PT_MODEL_CLASSES_MAPPING = {
    'decoder': AutoModelForCausalLM,
    't5': T5ForConditionalGeneration,
    'blenderbot': BlenderbotForConditionalGeneration,
    'mpt': AutoModelForCausalLM,
    'falcon': AutoModelForCausalLM,
    'stable_diffusion': DiffusionPipeline,
    'ldm_super_resolution': LDMSuperResolutionPipeline,
    'chatglm': AutoModel,
}

USE_CASES = {
    'image_gen': ['stable-diffusion-', 'deepfloyd-if'],
    'text2speech': ['whisper'],
    'image_cls': ['vit'],
    'code_gen': ['replit', 'codegen2', 'codegen', 'codet5'],
    'text_gen': ['decoder', 't5', 'falcon', 'gpt', 'aquila', 'mpt', 'open-llama', 'llama', 
                 'opt-', 'pythia-', 'stablelm-', 'blenderbot', 'vicuna', 'dolly', 'bloom', 
                 'red-pajama', 'chatglm', 'xgen', 'longchat', 'jais'],
    'ldm_super_resolution': ['ldm-super-resolution'],
}

DEFAULT_MODEL_CLASSES = {
    'text_gen': 'decoder',
    'image_gen': 'stable_diffusion',
    'image_cls': 'vit',
    'speech2text': 'whisper',
    'code_gen': 'decoder',
    'ldm_super_resolution': 'ldm_super_resolution',
}
