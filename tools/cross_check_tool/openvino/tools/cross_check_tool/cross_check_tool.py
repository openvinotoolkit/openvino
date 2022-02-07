#!/usr/bin/python3

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging as log
import sys
from collections import defaultdict
from typing import Union
import numpy as np

try:
    from openvino.runtime import Core, Model, CompiledModel, InferRequest, Output, Type, get_version
except Exception as e:
    exception_type = type(e).__name__
    print(f"The following error happened while importing OpenVINO Python API module:\n[ {exception_type} ] {e}")
    sys.exit(1)

from openvino.tools.cross_check_tool.utils import get_config_dictionary, get_layers_list, print_output_layers, \
    input_processing, accuracy_metrics, validate_args, build_parser, set_logger, find_out_cct_mode, \
    print_all_over_the_net_metrics, update_global_accuracy_matrics, blob_counters, performance_metrics, \
    dump_output_file, load_dump, error_handling, print_input_layers, set_verbosity, perf_counts_to_dump, load_profiling_info


###
#   PLUGIN
###


@error_handling('plugin of \'{device}\' device config \'{config}\' loading')
def set_plugin_config(core: Core, device: str, config: str = None):
    core.set_config(get_config_dictionary(config_file=config), device_name=device)


@error_handling('\'{cpu_ext}\' cpu extensions loading')
def set_cpu_extensions(core: Core, cpu_ext: str):
    core.add_extension(cpu_ext)


def get_plugin(device: str, cpu_ext: str = None, config: str = None):
    core = Core()
    # log.info('{} plugin:\n          API version ............ {}'.format(device, plugin.version), extra={'no_lvl': True})
    set_plugin_config(core=core, device=device, config=config)
    if cpu_ext and 'CPU' in device:
        set_cpu_extensions(core=core, cpu_ext=cpu_ext)
    return core


###
#   MODEL
###


@error_handling('reading {xml_path} IR model')
def get_model(model_path: str, core: Core):
    model = core.read_model(model=model_path)
    if model.is_dynamic():
        raise Exception("Cross check tool doesn't support dynamic models for now.")
    return model


@error_handling('compiling model for {device} device')
def get_compiled_model(core: Core, model: Model, device: str):
    return core.compile_model(model=model, device_name=device)


@error_handling('creating infer request')
def get_infer_request(compiled_model: CompiledModel):
    return compiled_model.create_infer_request()


@error_handling('output \'{output}\' addition for network from model \'{model}\'')
def get_model_copy_with_output(model: str, output: tuple, core: Core):
    model_copy = get_model(model_path=model, core=core)
    new_output = None
    if output not in ['None', None]:
        new_output = model_copy.add_outputs(output).pop()
    return model_copy, new_output


@error_handling('getting model layers info')
def get_model_info(model: Model):
    precision = None
    for input in model.inputs:
        if precision == None:
            precision = input.element_type
        elif precision != input.element_type:
            precision = Type.undefined
    return model.get_ordered_ops(), model.inputs, model.outputs, precision.get_type_name()


def check_inputs_and_default_outputs_are_equal(model, ref_model):
    if len(model.inputs) != len(ref_model.inputs):
        raise Exception("Models have different number of inputs! Cannot cross check!")
    if len(model.outputs) != len(ref_model.outputs):
        raise Exception("Models have different number of outputs! Cannot cross check!")
    for input, ref_input in zip(model.inputs, ref_model.inputs):
        if input.any_name != ref_input.any_name:
            raise Exception("Models have different inputs! Cannot cross check!")
    for output, ref_output in zip(model.outputs, ref_model.outputs):
        if output.any_name != ref_output.any_name:
            raise Exception("Models have different outputs! Cannot cross check!")


def get_layers_intersection(layers, ref_layers):
    layers_map = {node.friendly_name: node for node in layers}
    operation_names = set(layers_map.keys())
    ref_operation_names = set(node.friendly_name for node in ref_layers)
    intersection_names = operation_names.intersection(ref_operation_names)
    return [layers_map[intersection_name] for intersection_name in intersection_names]


def get_layers_union(layers, ref_layers):
    layers_map = {}
    for layer, ref_layer in zip(layers, ref_layers):
        layers_map.update({layer.friendly_name: layer})
        layers_map.update({ref_layer.friendly_name: ref_layer})
    return layers_map.values()

###
#   INFER
###


# TODO: replace ports by names in error_handing
@error_handling('getting inference results for output: \'{output.any_name}\'')
def get_infer_results(infer_request: InferRequest, output: Output):
    return infer_request.get_tensor(output).data


@error_handling('getting performance counts from infer request')
def get_profiling_info(infer_request: InferRequest, port: Output):
    for pi in infer_request.profiling_info:
       if pi.node_name == port.node.friendly_name:
           return pi


@error_handling('processing inference on \'{device}\' device')
def infer(model: Model, core: Core, device: str, inputs: Union[list, dict], output=None):
    compiled_model = get_compiled_model(core=core, model=model, device=device)
    infer_request = get_infer_request(compiled_model)
    infer_request.infer(inputs)
    if output:
        result = get_infer_results(infer_request, output)
        prof_info = get_profiling_info(infer_request, output)
        return result, prof_info


@error_handling('getting inference results for outputs: \'{layers}\'')
def overall_accuracy_check(model: str, ref_model: str, out_layers: list, ref_out_layers: list, inputs: list,
                           ref_inputs: list, core: Core, device: str, ref_core: Core, ref_device: str, layers: str,
                           num_of_iterations: int):
    global_times, ref_global_times = [], []
    if layers in ['None', None]:
        model_copy, _ = get_model_copy_with_output(model=model, output=layers, core=core)
        ref_model_copy, _ = get_model_copy_with_output(model=ref_model, output=layers, core=ref_core)
        for i in range(num_of_iterations):
            t1 = datetime.datetime.now()
            infer(model=model_copy, core=core, device=device, inputs=inputs)
            t2 = datetime.datetime.now()
            infer(model=ref_model_copy, core=ref_core, device=ref_device, inputs=ref_inputs)
            t3 = datetime.datetime.now()
            global_times.append(t2 - t1)
            ref_global_times.append(t3 - t2)
    return global_times, ref_global_times


def one_ir_mode(args):
    core = get_plugin(args.device, args.l, args.config)
    model = get_model(model_path=args.model, core=core)
    model_layers, model_inputs, model_outputs, precision = get_model_info(model)
    log.info(f'{args.device} : {precision} vs {args.reference_device} : {precision}')
    log.info(f'The same IR on both devices: {args.model}')
    out_layers = get_layers_list(model_layers, model_outputs, args.layers)
    print_input_layers(model_inputs)
    print_output_layers(out_layers)
    ref_core = get_plugin(args.reference_device, args.l, args.reference_config)
    global_accuracy = []
    inputs = input_processing(model_path=args.model, model_inputs=model_inputs, input_file=args.input)
    global_times, ref_global_times = overall_accuracy_check(model=args.model, ref_model=args.model,
                                                            out_layers=out_layers, ref_out_layers=out_layers,
                                                            inputs=inputs, ref_inputs=inputs, core=core,
                                                            device=args.device, ref_core=ref_core,
                                                            ref_device=args.reference_device, layers=args.layers,
                                                            num_of_iterations=args.num_of_iterations)
    for out_layer in out_layers:
        log.info(f'Layer {out_layer.friendly_name} statistics')
        for i in range(out_layer.get_output_size()):
            if out_layer.get_output_size() > 1:
                log.info(f'Port {i}: ')
            model_copy, new_output = get_model_copy_with_output(model=args.model, output=(out_layer.friendly_name, i), core=core)
            out_blob, pc = infer(model=model_copy, core=core, device=args.device, inputs=inputs, output=new_output)
            ref_out_blob, ref_pc = infer(model=model_copy, core=ref_core, device=args.reference_device, inputs=inputs, output=new_output)
            a_m = accuracy_metrics(out_blob=out_blob, ref_out_blob=ref_out_blob)
            performance_metrics(args.device, pc, args.reference_device, ref_pc)
            blob_counters(out_blob=out_blob, ref_out_blob=ref_out_blob)
            global_accuracy = update_global_accuracy_matrics(global_accuracy=global_accuracy, current_accuracy=a_m)
    print_all_over_the_net_metrics(global_times=global_times, ref_global_times=ref_global_times,
                                   global_accuracy=global_accuracy)


def two_ir_mode(args):
    core = get_plugin(args.device, args.l, args.config)
    ref_core = get_plugin(args.reference_device, args.l, args.reference_config)
    model = get_model(model_path=args.model, core=core)
    model_layers, model_inputs, model_outputs, precision = get_model_info(model)
    ref_model = get_model(model_path=args.reference_model, core=ref_core)
    ref_model_layers, _, _, ref_precision = get_model_info(ref_model)
    check_inputs_and_default_outputs_are_equal(model, ref_model)
    log.info(f'{args.device} : {precision} vs {args.reference_device} : {ref_precision}')
    log.info(f'IR for {args.device} : {args.model}')
    log.info(f'IR for {args.reference_device} : {args.reference_model}')
    if args.reference_layers:
        out_layers = get_layers_list(model_layers, model_outputs, args.layers)
        ref_out_layers = get_layers_list(ref_model_layers, model_outputs, args.reference_layers)
        if len(out_layers) != len(ref_out_layers):
            raise Exception("Number of layers to compare against should be equal!")
    else:
        ref_out_layers = out_layers = get_layers_list(get_layers_intersection(model_layers, ref_model_layers), model_outputs, args.layers)
    print_input_layers(model_inputs)
    print_output_layers(get_layers_union(out_layers, ref_out_layers))
    inputs = input_processing(model_path=args.model, model_inputs=model_inputs, input_file=args.input)
    global_accuracy = []
    global_times, ref_global_times = overall_accuracy_check(model=args.model, ref_model=args.reference_model,
                                                            out_layers=out_layers, ref_out_layers=out_layers,
                                                            inputs=inputs, ref_inputs=inputs, core=core,
                                                            device=args.device, ref_core=ref_core,
                                                            ref_device=args.reference_device, layers=args.layers,
                                                            num_of_iterations=args.num_of_iterations)
    for out_layer, ref_out_layer in zip(out_layers, ref_out_layers):
        if out_layer.friendly_name == ref_out_layer.friendly_name:
            log.info(f'Layer {out_layer.friendly_name} statistics')
        else:
            if out_layer.get_output_size() != ref_out_layer.get_output_size():
                log.warning(f"Skipping {out_layer.friendly_name} vs {ref_out_layer.frinedly_name} comparison due to different number of outputs!")
                continue
            log.info(f'Layer {out_layer.friendly_name} vs {ref_out_layer.friendly_name} statistics')
        for i in range(out_layer.get_output_size()):
            if out_layer.get_output_size() > 1:
                log.info(f'Port {i}: ')
            model_copy, new_output = get_model_copy_with_output(model=args.model, output=(out_layer.friendly_name, i), core=core)
            ref_model_copy, ref_new_output = get_model_copy_with_output(model=args.reference_model, output=(ref_out_layer.friendly_name, i), core=ref_core)
            out_blob, pc = infer(model=model_copy, core=core, device=args.device, inputs=inputs, output=new_output)
            ref_out_blob, ref_pc = infer(model=ref_model_copy, core=ref_core, device=args.reference_device,
                                                                    inputs=inputs, output=ref_new_output)
            a_m = accuracy_metrics(out_blob=out_blob, ref_out_blob=ref_out_blob)
            performance_metrics(args.device, pc, args.reference_device, ref_pc)
            blob_counters(out_blob=out_blob, ref_out_blob=ref_out_blob)
            global_accuracy = update_global_accuracy_matrics(global_accuracy=global_accuracy, current_accuracy=a_m)
    print_all_over_the_net_metrics(global_times=global_times, ref_global_times=ref_global_times,
                                   global_accuracy=global_accuracy)


def dump_mode(args):
    core = get_plugin(args.device, args.l, args.config)
    model = get_model(model_path=args.model, core=core)
    model_layers, model_inputs, model_outputs, _ = get_model_info(model)
    out_layers = get_layers_list(model_layers, model_outputs, args.layers)
    inputs = input_processing(args.model, model_inputs, args.input)
    dump_dict = defaultdict(list)
    for out_layer in out_layers:
        for i in range(out_layer.get_output_size()):
            if out_layer.get_output_size() > 1:
                log.info(f'Layer {out_layer.friendly_name}, port {i} processing')
            else:
                log.info(f'Layer {out_layer.friendly_name} processing')
            model_copy, new_output = get_model_copy_with_output(model=args.model, output=(out_layer.friendly_name, i), core=core)
            out_blob, pc = infer(model=model_copy, core=core, device=args.device, inputs=inputs, output=new_output)
            dump_dict[out_layer.friendly_name].append(np.array({'blob': out_blob, 'pc': perf_counts_to_dump(pc)}))
    dump_dict["device"] = args.device
    dump_output_file(args.model + '_' + args.device + '_dump.npz', dump_dict)


def load_mode(args):
    core = get_plugin(args.device, args.l, args.config)
    log.info(f'IR for {args.device} : {args.model}')
    log.info(f'Loading blob from {args.load}')
    model = get_model(model_path=args.model, core=core)
    model_layers, model_inputs, model_outputs, _ = get_model_info(model)
    out_layers = get_layers_list(model_layers, model_outputs, args.layers)
    print_input_layers(model_inputs)
    print_output_layers(out_layers)
    inputs = input_processing(args.model, model_inputs, args.input)
    global_accuracy = []
    loaded = load_dump(args.load)
    for out_layer in out_layers:
        if out_layer.friendly_name in loaded:
            log.info(f'Layer {out_layer.friendly_name} statistics')
        else:
            log.info(f'Statistics for layer \'{out_layer.friendly_name}\' was not dumped. Skipping this layer.')
            continue
        for i in range(out_layer.get_output_size()):
            if out_layer.get_output_size() > 1:
                log.info(f'Port {i}: ')
            model_copy, new_output = get_model_copy_with_output(model=args.model, output=(out_layer.friendly_name, i), core=core)
            out_blob, pc = infer(model=model_copy, core=core, device=args.device, inputs=inputs, output=new_output)
            ref_out_blob, ref_pc = loaded[out_layer.friendly_name][i]['blob'], load_profiling_info(loaded[out_layer.friendly_name][i]['pc'])
            a_m = accuracy_metrics(out_blob=out_blob, ref_out_blob=ref_out_blob)
            performance_metrics(args.device, pc, loaded["device"], ref_pc)
            blob_counters(out_blob=out_blob, ref_out_blob=ref_out_blob)
            global_accuracy = update_global_accuracy_matrics(global_accuracy=global_accuracy, current_accuracy=a_m)
    print_all_over_the_net_metrics(global_accuracy=global_accuracy)


def main():
    set_logger(log.DEBUG)
    args = validate_args(build_parser().parse_args())

    log.info(f'OpenVINO:\n          API version ............ {get_version()}', extra={'no_lvl': True})
    set_verbosity(args.verbosity)
    mode = find_out_cct_mode(args)
    if mode == 1:
        log.info('Cross check with one IR was enabled')
        one_ir_mode(args)
    elif mode == 2:
        log.info('Cross check with two IRs was enabled')
        two_ir_mode(args)
    elif mode == 3:
        log.info('Dump mode was enabled')
        dump_mode(args)
    elif mode == 4:
        log.info('Load mode was enabled')
        load_mode(args)
    log.info("Execution successful")


if __name__ == '__main__':
    main()
