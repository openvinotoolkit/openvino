# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from transformers import AutoConfig
from openvino.tools import mo
from openvino.runtime import serialize, Core
import openvino as ov
import torch
import timm
import time
import types

from utils.config_class import OV_MODEL_CLASSES_MAPPING, TOKENIZE_CLASSES_MAPPING, DEFAULT_MODEL_CLASSES
from utils.model_utils import get_config

from transformers.modeling_outputs import CausalLMOutputWithPast
from openvino import Type, Tensor
import numpy as np


def forward_simplified(
    self,
    input_ids: torch.LongTensor,
    attention_mask = None,
    past_key_values = None,
    **kwargs,
) -> CausalLMOutputWithPast:
    self.compile()

    if self.use_cache and past_key_values is not None:
        input_ids = input_ids[:, -1:]

    inputs = {}
    if not self.use_cache_as_state:
        if past_key_values is not None:
            if self._pkv_precision == Type.bf16:
                # numpy does not support bf16, pretending f16, should change to bf16
                past_key_values = tuple(
                    Tensor(past_key_value, past_key_value.shape, Type.bf16)
                    for pkv_per_layer in past_key_values
                    for past_key_value in pkv_per_layer
                )
            else:
                # Flatten the past_key_values
                past_key_values = tuple(
                    past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
                )
            # Add the past_key_values to the decoder inputs
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        # Create empty past_key_values for decoder_with_past first generation step
        elif self.use_cache:
            shape_input_ids = input_ids.shape
            num_attention_heads = (
                self.normalized_config.num_attention_heads if self.config.model_type == "bloom" else 1
            )
            for input_name in self.key_value_input_names:
                model_inputs = self.model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = shape_input_ids[0] * num_attention_heads
                if shape[2].is_dynamic:
                    shape[2] = 0
                if shape[1].is_dynamic:
                    shape[1] = 0
                if shape.rank.get_length() == 4 and shape[3].is_dynamic:
                    shape[3] = 0
                inputs[input_name] = Tensor(model_inputs.get_element_type(), shape.get_shape())
    else:
        # past_key_values are not used explicitly, instead they should be handled inside the model
        if past_key_values is None:
            past_key_values = ()   # something that is not None to differentiate the first iteration in the first condition in this function
            # TODO: resent state?

    inputs["input_ids"] = np.array(input_ids)

    # Add the attention_mask inputs when needed
    if "attention_mask" in self.input_names and attention_mask is not None:
        inputs["attention_mask"] = np.array(attention_mask)

    if hasattr(self, 'next_beam_idx'):
        inputs['beam_idx'] = np.array(self.next_beam_idx)

    # Run inference
    self.request.start_async(inputs, share_inputs=True)
    self.request.wait()

    # this is probably not real logits but already post-processed values depending on whether post-processing is fused into a model or not
    logits = torch.from_numpy(self.request.get_tensor("logits").data).to(self.device)

    if not self.use_cache_as_state:
        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(
                past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
            )
        else:
            past_key_values = None

    return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


def generate_simplified(self, *args, **kwargs):
    if len(args):
        raise Exception(f'Not empty args is not supported in generate_simplified, given: {args}')
    # TODO: Check other ignored parameters and report about them

    print('[ WARNING ] Termination criteria is not supported in overridden generate, max_new_tokens only matters')

    # TODO: Check if unsupported kwargs are provided

    input_ids = kwargs['input_ids']
    attention_mask = kwargs['attention_mask']

    assert kwargs['num_beams'] == 1, "Overridden generate doesn't support num_beams > 1"

    past_key_values = None

    for i in range(kwargs['max_new_tokens']):
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)

        next_tokens = outputs.logits   # logits is an old name from original model, when interprocessing is fused it is a token
        # TODO: Apply termination criteria in addition to max_new_tokens
        # TODO: Doing the cat with input_ids here, we will 'uncat' it later in the next forward, avoid doing it by passible next_tokens (without cat) directly to the next forward
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        # Depending on whether we applied make_stateful, past_key_values may or may not represent meaningful values, need to pass them anyway to differentiate the first iteration
        past_key_values = outputs.past_key_values

    return input_ids


def patch_inter_processing(hf_model, **kwargs):
    '''Fuse post-processing as an extra ops into a model'''

    ov_model = hf_model.model

    if kwargs['fuse_decoding_strategy']:
        ppp = ov.preprocess.PrePostProcessor(ov_model)
        import openvino.runtime.opset12 as opset

        assert kwargs['num_beams'] == 1, "Parameter fuse_decoding_strategy doesn't support beam_search, set num_beams to 1"

        def greedy_search(input):
            next_token = opset.gather(input, opset.constant(-1), opset.constant(1))  # take last logits only (makes sense at the first iteration only)
            topk = opset.topk(next_token, opset.constant(1), axis=-1, mode='max', sort='none').output(1)
            return topk
        ppp.output(0).postprocess().custom(greedy_search)

        ov_model = ppp.build()
        hf_model.model = ov_model
        hf_model._orig_generate = hf_model.generate
        hf_model.generate = types.MethodType(generate_simplified, hf_model)

    num_beams = kwargs['num_beams']

    if kwargs['fuse_cache_reorder'] and num_beams > 1:
        # Should be run before make_stateful because of adding pre-processing on kv-cashe inputs
        # Make a new parameter for beam_idx
        import openvino.runtime.opset12 as opset
        # Adding a new parameter to make _reorder_cache inside the model in the beginning of each iteration
        beam_idx = opset.parameter(name='beam_idx', dtype=ov.Type.i32, shape=ov.PartialShape([num_beams]))
        beam_idx.output(0).get_tensor().add_names({'beam_idx'})   # why list is not accepted?
        ov_model.add_parameters([beam_idx])
        # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
        for i in range(len(ov_model.inputs) - 3):   # 3 == input_ids, attention_mask and new beam_idx
            parameter_output_port = ov_model.inputs[2 + i]
            consumers = parameter_output_port.get_target_inputs()
            gather = opset.gather(parameter_output_port, beam_idx, opset.constant(0))
            for consumer in consumers:
                consumer.replace_source_output(gather.output(0))
        ov_model.validate_nodes_and_infer_types()
        hf_model.use_cache_as_state = False

        # override _reorder_cache to avoid cache manipulation outside of the model as it is already done inside
        def _reorder_cache_stub(self, past_key_values, beam_idx):
            self.next_beam_idx = np.array(beam_idx)   # save beam_idx to be used as an input in the next iteration
            return past_key_values

        hf_model._reorder_cache = types.MethodType(_reorder_cache_stub, hf_model)
        hf_model.forward = types.MethodType(forward_simplified, hf_model)   # need custom forward to set beam_idx input to OV model
        hf_model.next_beam_idx = np.zeros([num_beams], dtype=int)    # initial value for beam_idx is all zeros

    if kwargs['make_stateful']:
        from openvino._offline_transformations import apply_make_stateful_transformation
        input_output_map = {}
        # TODO: Can we derive the dimensions from the model topology?
        num_attention_heads = (
            hf_model.normalized_config.num_attention_heads if hf_model.config.model_type == "bloom" else 1
        )
        num_beams = kwargs['num_beams'] if 'num_beams' in kwargs and kwargs['num_beams'] > 1 else 1
        beam_idx_exist = 'beam_idx' in [input.any_name for input in ov_model.inputs]
        assert num_beams == 1 or beam_idx_exist, 'Requested to make_stateful with num_beams > 1 but there is no beam_idx parameter for cache reorder fused'
        left_num_parameters = 2 + int(beam_idx_exist)
        for i in range(len(ov_model.inputs) - left_num_parameters):
            input = ov_model.inputs[2 + i]
            output = ov_model.outputs[1 + i]
            input_output_map[input.any_name] = output.any_name
            shape = input.get_partial_shape()

            # suppose 0-th dimension is a batch
            # TODO: Deduce from a model via ordinal reshape
            shape[0] = kwargs['batch_size'] * num_attention_heads * num_beams

            input.get_node().set_partial_shape(shape)

        ov_model.validate_nodes_and_infer_types()

        apply_make_stateful_transformation(ov_model, input_output_map)

        hf_model.use_cache_as_state = True
        hf_model.forward = types.MethodType(forward_simplified, hf_model)  # override to avoid cache manipulation outside of the model

    xml_file_name = kwargs['save_prepared_model']
    if xml_file_name is not None:
        print(f'Saving prepared OpenVINO model to {xml_file_name} ...')
        ov.save_model(ov_model, xml_file_name)

    hf_model.compile()


def create_text_gen_model(model_path, device, **kwargs):
    '''
    - model_path: can be model_id, model_path or IR path
    - device: can be CPU or GPU
    - model_type:
    '''
    default_model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
    model_type = kwargs.get("model_type" , default_model_type)
    model_class = OV_MODEL_CLASSES_MAPPING.get(model_type, OV_MODEL_CLASSES_MAPPING[default_model_type])
    token_class = TOKENIZE_CLASSES_MAPPING.get(model_type, TOKENIZE_CLASSES_MAPPING[default_model_type])
    model_path = Path(model_path)
    # specify the model path
    if model_path.name.endswith("xml"):
        model_path = model_path.parents[2]

    ov_config = kwargs["config"]

    model_path_existed = Path(model_path).exists()
    # load model
    if not model_path_existed:
        if model_type in ["mpt", "falcon", "replit", "codegen2", "chatglm", "chatglm2"]:
            start = time.perf_counter()
            ov_model = model_class.from_pretrained(kwargs['model_id'], device=device, export=True, ov_config=ov_config,
            config=AutoConfig.from_pretrained(kwargs['model_id'], trust_remote_code=True))
            end = time.perf_counter()
        else:
            start = time.perf_counter()
            ov_model = model_class.from_pretrained(kwargs['model_id'], export=True, device=device, ov_config=ov_config)
            end = time.perf_counter()
        ov_model.save_pretrained(model_path)
    else:
        if model_type in ["mpt", "falcon", "replit", "codegen2", "chatglm", 'chatglm2']:
            start = time.perf_counter()
            ov_model = model_class.from_pretrained(
                model_path, device=device, ov_config=ov_config,
                config=AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                )
            end = time.perf_counter()
        else:
            start = time.perf_counter()
            ov_model = model_class.from_pretrained(model_path, device=device, ov_config=ov_config, compile=False)
            patch_inter_processing(ov_model, **kwargs)
            end = time.perf_counter()
    from_pretrained_time = end - start
    print(f"from pretrained time: {from_pretrained_time:.2f}s")
    # load token
    if not model_path_existed:
        tokenizer = token_class.from_pretrained(kwargs['model_id'])
        tokenizer.save_pretrained(model_path)
    else:
        tokenizer = token_class.from_pretrained(model_path, trust_remote_code=True)
    return ov_model, tokenizer, from_pretrained_time


def create_image_gen_model(model_path, device, **kwargs):
    model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
    model_class = OV_MODEL_CLASSES_MAPPING[model_type]
    model_path = Path(model_path)
    ov_config = kwargs["config"]
    if not Path(model_path).exists():
        start = time.perf_counter()
        ov_model = model_class.from_pretrained(kwargs['model_id'], from_transformers=True, device=device, ov_config=ov_config)
        end = time.perf_counter()
        ov_model.save_pretrained(model_path)
    else:
        print(f"model_path={model_path}")
        start = time.perf_counter()
        ov_model = model_class.from_pretrained(model_path, device=device, ov_config=ov_config)
        end = time.perf_counter()
    from_pretrained_time = end - start
    print(f"from pretrained time: {from_pretrained_time:.2f}s")
    return ov_model, from_pretrained_time


def create_image_classification_model(model_path, device, config=None, **kwargs):
    core = Core()
    if config is not None:
        ov_config = get_config(config)
        core.set_property(ov_config)
    model_path = Path(model_path)
    model_file = None
    if model_path.exists():
        if model_path.is_dir():
            model_file = list(model_path.rglob("*.xml"))
            if model_file:
                model_file = model_file[0]
        else:
            model_file = model_path
    model_id =  model_path.name
    data_config = timm.data.resolve_data_config([], model=model_id, use_test_size=True)
    input_size = data_config["input_size"]
    input_size = (1, ) + input_size
    if model_file is None:
        pt_model = timm.create_model(model_id, pretrained=True)
        ov_model = mo.convert_model(pt_model, example_input=torch.randn(input_size))
        serialize(ov_model, str(model_path / "dldt" / "FP32" / (model_path.name + ".xml")))
    else:
        start = time.perf_counter()
        ov_model = core.read_model(model_file)
        end = time.perf_counter()
        load_model_time = end - start
        print(f"load model time: {load_model_time:.2f}s")
    start = time.perf_counter()
    compiled_model = core.compile_model(ov_model, device.upper())
    end = time.perf_counter()
    compile_model_time = end - start
    print(f"compile model time: {compile_model_time:.2f}s")
    return compiled_model, input_size


def create_ldm_super_resolution_model(model_path, device, **kwargs):
    core = Core()
    ov_config = kwargs["config"]
    core.set_property(ov_config)
    default_model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
    model_type = kwargs.get("model_type" , default_model_type)
    model_class = OV_MODEL_CLASSES_MAPPING[model_type]
    model_path = Path(model_path)
    start = time.perf_counter()
    ov_model = model_class(model_path, core, device.upper())
    end = time.perf_counter()
    from_pretrained_time = end - start
    print(f"from pretrained time: {from_pretrained_time:.2f}s")
    return ov_model, from_pretrained_time
