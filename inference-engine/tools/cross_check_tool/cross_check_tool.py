# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import datetime
import logging as log
import os
import sys

import numpy as np

try:
    from openvino import inference_engine as ie
    from openvino.inference_engine import IENetwork, IECore
except Exception as e:
    exception_type = type(e).__name__
    print("The following error happened while importing Python API module:\n[ {} ] {}".format(exception_type, e))
    sys.exit(1)

try:
    import ngraph as ng
except Exception as e:
    exception_type = type(e).name
    print("The following error happened while importing nGraph module:\n[ {} ] {}".format(exception_type, e))
    sys.exit(1)

from utils import get_config_dictionary, get_layers_list, print_output_layers, input_processing, \
    accuracy_metrics, validate_args, build_parser, set_logger, find_out_cct_mode, print_all_over_the_net_metrics, \
    update_global_accuracy_matrics, blob_counters, performance_metrics, manage_user_outputs_with_mapping, \
    dump_output_file, load_dump, error_handling, print_input_layers, set_verbosity


###
#   PLUGIN
###


@error_handling('plugin of \'{plugin.device}\' device config \'{config}\' loading')
def set_plugin_config(core: IECore, device: str, config: str = None):
    core.set_config(get_config_dictionary(config_file=config), device_name=device)


@error_handling('\'{cpu_ext}\' cpu extensions loading')
def set_cpu_extensions(core: IECore, cpu_ext: str):
    core.add_extension(cpu_ext, "CPU")


def get_plugin(device: str, cpu_ext: str = None, config: str = None):
    ie = IECore()
    # log.info('{} plugin:\n          API version ............ {}'.format(device, plugin.version), extra={'no_lvl': True})
    set_plugin_config(core=ie, device=device, config=config)
    if cpu_ext and 'CPU' in device:
        set_cpu_extensions(core=ie, cpu_ext=cpu_ext)
    return ie


###
#   MODEL
###


@error_handling('reading {model} IR model')
def get_net(model: str, core: IECore):
    model_xml = model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    net = core.read_network(model=model_xml, weights=model_bin)
    return net


@error_handling('loading network to plugin of {plugin.device} device')
def get_exec_net(core, net, device):
    return core.load_network(network=net, device_name=device)


@error_handling('output \'{output}\' addition for network from model \'{model}\'')
def get_net_copy_with_output(model: str, output: str, core: IECore):
    net_copy = get_net(model=model, core=core)
    if output not in ['None', None]:
        net_copy.add_outputs(output)
    return net_copy


@error_handling('getting model layers info')
def get_model_info(net: IENetwork):
    func = ng.function_from_cnn(net)
    ops = func.get_ordered_ops()
    return ops, net.input_info, net.outputs


###
#   INFER
###


@error_handling('processing inference on \'{device}\' device')
def get_infer_results(executable_network, inputs: dict):
    return executable_network.infer(inputs=inputs)


@error_handling('getting performance counts from executable network on \'{device}\' device')
def get_perf_counts(executable_network):
    return executable_network.requests[0].get_perf_counts()


@error_handling('getting inference results for outputs: \'{output}\'')
def infer(net: IENetwork, core: IECore, device: str, inputs: dict, output: list):
    executable_network = get_exec_net(core=core, net=net, device=device)
    infer_dict = get_infer_results(executable_network=executable_network, inputs=inputs)
    pc = get_perf_counts(executable_network=executable_network)
    no_i = 'no_info'
    no_info_pc = {'cpu_time': no_i, 'exec_time': no_i, 'layer_type': no_i, 'real_time': no_i, 'status': no_i}
    result = {}
    for out in output:
        if out not in infer_dict:
            log.warning("There is no '{}' layer in Inference Engine outputs results".format(out))
            continue
        pc = pc[out] if out in pc else no_info_pc
        pc['device'] = device
        result = {out: [infer_dict[out], pc]}
    return result


@error_handling('getting inference results for outputs: \'{output}\'')
def overall_accuracy_check(model: str, ref_model: str, out_layers: list, ref_out_layers: list, inputs: dict,
                           ref_inputs: dict, core: IECore, device: str, ref_core: IECore, ref_device: str, layers: str,
                           num_of_iterations: int):
    global_times, ref_global_times = [], []
    if layers in ['None', None]:
        net_copy = get_net_copy_with_output(model=model, output=layers, core=core)
        ref_net_copy = get_net_copy_with_output(model=ref_model, output=layers, core=ref_core)
        for i in range(num_of_iterations):
            t1 = datetime.datetime.now()
            infer(net=net_copy, core=core, device=device, inputs=inputs, output=out_layers)
            t2 = datetime.datetime.now()
            infer(net=ref_net_copy, core=ref_core, device=ref_device, inputs=ref_inputs, output=ref_out_layers)
            t3 = datetime.datetime.now()
            global_times.append(t2 - t1)
            ref_global_times.append(t3 - t2)
    return global_times, ref_global_times


def one_ir_mode(args):
    core = get_plugin(args.device, args.l, args.config)
    net = get_net(model=args.model, core=core)
    net_layers, net_inputs, net_outputs = get_model_info(net)
    log.info('{} vs {}'.format(args.device, args.reference_device))
    log.info('The same IR on both devices: {}'.format(args.model))
    out_layers = get_layers_list(net_layers, net_inputs, net_outputs, args.layers)
    print_input_layers(net_inputs)
    print_output_layers(out_layers)
    ref_core = get_plugin(args.reference_device, args.l, args.reference_config)
    global_accuracy = []
    inputs = input_processing(model_path=args.model, net_inputs=net_inputs, input_file=args.input)
    global_times, ref_global_times = overall_accuracy_check(model=args.model, ref_model=args.model,
                                                            out_layers=out_layers, ref_out_layers=out_layers,
                                                            inputs=inputs, ref_inputs=inputs, core=core,
                                                            device=args.device, ref_core=ref_core,
                                                            ref_device=args.reference_device, layers=args.layers,
                                                            num_of_iterations=args.num_of_iterations)
    for out_layer in out_layers:
        log.info('Layer {} statistics'.format(out_layer))
        net_copy = get_net_copy_with_output(model=args.model, output=out_layer, core=core)
        results = infer(net=net_copy, core=core, device=args.device, inputs=inputs, output=[out_layer])
        if out_layer not in results:
            continue
        out_blob, pc = results[out_layer]
        ref_results = infer(net=net_copy, core=ref_core, device=args.reference_device,
                            inputs=inputs, output=[out_layer])
        if out_layer not in ref_results:
            continue
        ref_out_blob, ref_pc = ref_results[out_layer]
        a_m = accuracy_metrics(out_blob=out_blob, ref_out_blob=ref_out_blob)
        performance_metrics(pc=pc, ref_pc=ref_pc)
        blob_counters(out_blob=out_blob, ref_out_blob=ref_out_blob)
        global_accuracy = update_global_accuracy_matrics(global_accuracy=global_accuracy, current_accuracy=a_m)
    print_all_over_the_net_metrics(global_times=global_times, ref_global_times=ref_global_times,
                                   global_accuracy=global_accuracy)


def two_ir_mode(args):
    core = get_plugin(args.device, args.l, args.config)
    ref_core = get_plugin(args.reference_device, args.l, args.reference_config)
    net = get_net(model=args.model, core=core)
    net_layers, net_inputs, net_outputs = get_model_info(net)
    ref_net = get_net(model=args.reference_model, core=ref_core)
    ref_net_layers, ref_net_inputs, ref_net_outputs = get_model_info(ref_net)
    log.info('{} vs {}'.format(args.device, args.reference_device))
    log.info('IR for {} : {}'.format(args.device, args.model))
    log.info('IR for {} : {}'.format(args.reference_device, args.reference_model))
    out_layers = get_layers_list(net_layers, net_inputs, net_outputs, args.layers)
    ref_out_layers = get_layers_list(ref_net_layers, ref_net_inputs, ref_net_outputs, args.layers)
    print_input_layers(net_inputs)
    print_output_layers(out_layers)
    layers_map = manage_user_outputs_with_mapping(mapping=args.mapping, reference_mapping=args.reference_mapping,
                                                  user_layers=out_layers)
    inputs = input_processing(model_path=args.model, net_inputs=net_inputs, input_file=args.input,
                              layers_map=layers_map)
    ref_inputs = input_processing(model_path=args.reference_model, net_inputs=ref_net_inputs, input_file=args.input,
                                  layers_map=layers_map)
    global_accuracy = []
    global_times, ref_global_times = overall_accuracy_check(model=args.model, ref_model=args.reference_model,
                                                            out_layers=out_layers, ref_out_layers=ref_out_layers,
                                                            inputs=inputs, ref_inputs=ref_inputs, plugin=core,
                                                            ref_plugin=ref_core, layers=args.layers,
                                                            num_of_iterations=args.num_of_iterations)
    for out_layer in layers_map:
        ref_out_layer = layers_map[out_layer]
        if out_layer == ref_out_layer:
            log.info('Layer {} statistics'.format(out_layer))
        else:
            log.info('Statistics \'{}\' vs \'{}\''.format(out_layer, ref_out_layer))
        net_copy = get_net_copy_with_output(model=args.model, output=out_layer, core=core)
        ref_net_copy = get_net_copy_with_output(model=args.reference_model, output=ref_out_layer, core=ref_core)
        results = infer(net=net_copy, core=core, device=args.device, inputs=inputs, output=[out_layer])
        if out_layer not in results:
            continue
        out_blob, pc = results[out_layer]
        ref_results = infer(net=ref_net_copy, core=ref_core, device=args.reference_device,
                            inputs=ref_inputs, output=[ref_out_layer])
        ref_out_blob, ref_pc = ref_results[ref_out_layer]
        if ref_out_layer not in ref_results:
            continue
        a_m = accuracy_metrics(out_blob=out_blob, ref_out_blob=ref_out_blob)
        performance_metrics(pc=pc, ref_pc=ref_pc)
        blob_counters(out_blob=out_blob, ref_out_blob=ref_out_blob)
        global_accuracy = update_global_accuracy_matrics(global_accuracy=global_accuracy, current_accuracy=a_m)
    print_all_over_the_net_metrics(global_times=global_times, ref_global_times=ref_global_times,
                                   global_accuracy=global_accuracy)


def dump_mode(args):
    core = get_plugin(args.device, args.l, args.config)
    net = get_net(model=args.model, core=core)
    func = ng.function_from_cnn(net)
    ops = func.get_ops()
    out_layers = get_layers_list(ops, net.input_info, net.outputs, args.layers)
    inputs = input_processing(args.model, net.input_info, args.input)
    dump_dict = {}
    for out_layer in out_layers:
        log.info('Layer {} processing'.format(out_layer))
        net_copy = get_net_copy_with_output(model=args.model, output=out_layer, core=core)
        results = infer(net=net_copy, core=core, device=args.device, inputs=inputs, output=[out_layer])
        if out_layer not in results:
            continue
        out_blob, pc = results[out_layer]
        dump_dict[out_layer] = np.array({'blob': out_blob, 'pc': pc})
    dump_output_file(args.model + '_' + args.device + '_dump.npz', dump_dict)


def load_mode(args):
    core = get_plugin(args.device, args.l, args.config)
    log.info('IR for {} : {}'.format(args.device, args.model))
    log.info('Loading blob from {}'.format(args.load))
    net = get_net(model=args.model, core=core)
    net_layers, net_inputs, net_outputs = get_model_info(net)
    out_layers = get_layers_list(net_layers, net_inputs, net_outputs, args.layers)
    print_input_layers(net_inputs)
    print_output_layers(out_layers)
    layers_map = manage_user_outputs_with_mapping(mapping=args.mapping, reference_mapping=args.reference_mapping,
                                                  user_layers=out_layers)
    inputs = input_processing(args.model, net_inputs, args.input, layers_map)
    global_accuracy = []
    loaded = load_dump(args.load)
    for out_layer in layers_map:
        ref_out_layer = layers_map[out_layer]
        if out_layer == ref_out_layer:
            log.info('Layer {} statistics'.format(out_layer))
        else:
            log.info('Statistics \'{}\' vs \'{}\''.format(out_layer, ref_out_layer))
        net_copy = get_net_copy_with_output(model=args.model, output=out_layer, core=core)
        results = infer(net=net_copy, core=core, device=args.device, inputs=inputs, output=[out_layer])
        if out_layer not in results:
            continue
        out_blob, pc = results[out_layer]
        if ref_out_layer not in loaded:
            continue
        ref_out_blob = loaded[ref_out_layer]['blob']
        a_m = accuracy_metrics(out_blob=out_blob, ref_out_blob=ref_out_blob)
        if 'pc' in loaded[ref_out_layer]:
            ref_pc = loaded[ref_out_layer]['pc']
            performance_metrics(pc=pc, ref_pc=ref_pc)
        blob_counters(out_blob=out_blob, ref_out_blob=ref_out_blob)
        global_accuracy = update_global_accuracy_matrics(global_accuracy=global_accuracy, current_accuracy=a_m)
    print_all_over_the_net_metrics(global_accuracy=global_accuracy)


def main(args):
    log.info('Inference Engine:\n          API version ............ {}'.format(ie.__version__), extra={'no_lvl': True})
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
    set_logger(log.DEBUG)
    main(validate_args(build_parser().parse_args()))
