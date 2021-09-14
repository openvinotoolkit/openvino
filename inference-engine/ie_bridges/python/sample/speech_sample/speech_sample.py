#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging as log
import re
import sys
from timeit import default_timer
from typing import Union

import numpy as np
from arg_parser import parse_args
from file_options import read_utterance_file, write_utterance_file
from openvino.inference_engine import ExecutableNetwork, IECore, IENetwork

# Operating Frequency for GNA HW devices for Core and Atom architecture
GNA_CORE_FREQUENCY = 400
GNA_ATOM_FREQUENCY = 200


def get_scale_factor(matrix: np.ndarray) -> float:
    """Get scale factor for quantization using utterance matrix"""
    # Max to find scale factor
    target_max = 16384
    max_val = np.max(matrix)
    if max_val == 0:
        return 1.0
    else:
        return target_max / max_val


def infer_data(data: dict, exec_net: ExecutableNetwork, input_blobs: list, output_blobs: list) -> np.ndarray:
    """Do a synchronous matrix inference"""
    matrix_shape = next(iter(data.values())).shape
    result = {}

    for blob_name in output_blobs:
        batch_size, num_of_dims = exec_net.outputs[blob_name].shape
        result[blob_name] = np.ndarray((matrix_shape[0], num_of_dims))

    slice_begin = 0
    slice_end = batch_size

    while slice_begin < matrix_shape[0]:
        vectors = {blob_name: data[blob_name][slice_begin:slice_end] for blob_name in input_blobs}
        num_of_vectors = next(iter(vectors.values())).shape[0]

        if num_of_vectors < batch_size:
            temp = {blob_name: np.zeros((batch_size, vectors[blob_name].shape[1])) for blob_name in input_blobs}

            for blob_name in input_blobs:
                temp[blob_name][:num_of_vectors] = vectors[blob_name]

            vectors = temp

        vector_results = exec_net.infer(vectors)

        for blob_name in output_blobs:
            result[blob_name][slice_begin:slice_end] = vector_results[blob_name][:num_of_vectors]

        slice_begin += batch_size
        slice_end += batch_size

    return result


def compare_with_reference(result: np.ndarray, reference: np.ndarray):
    error_matrix = np.absolute(result - reference)

    max_error = np.max(error_matrix)
    sum_error = np.sum(error_matrix)
    avg_error = sum_error / error_matrix.size
    sum_square_error = np.sum(np.square(error_matrix))
    avg_rms_error = np.sqrt(sum_square_error / error_matrix.size)
    stdev_error = np.sqrt(sum_square_error / error_matrix.size - avg_error * avg_error)

    log.info(f'max error: {max_error:.7f}')
    log.info(f'avg error: {avg_error:.7f}')
    log.info(f'avg rms error: {avg_rms_error:.7f}')
    log.info(f'stdev error: {stdev_error:.7f}')


def get_input_layer_list(net: Union[IENetwork, ExecutableNetwork], args: argparse.Namespace) -> list:
    """Get a list of input layer names"""
    return re.split(', |,', args.input_layers) if args.input_layers else [next(iter(net.input_info))]


def get_output_layer_list(net: Union[IENetwork, ExecutableNetwork],
                          args: argparse.Namespace, with_ports: bool) -> list:
    """Get a list of output layer names"""
    if args.output_layers:
        output_name_port = [output.split(':') for output in re.split(', |,', args.output_layers)]
        if with_ports:
            try:
                return [(blob_name, int(port)) for blob_name, port in output_name_port]
            except ValueError:
                log.error('Incorrect value for -oname/--output_layers option, please specify a port for output layer.')
                sys.exit(-4)
        else:
            return [blob_name for blob_name, _ in output_name_port]
    else:
        return [list(net.outputs.keys())[-1]]


def parse_scale_factors(args: argparse.Namespace) -> list:
    """Get a list of scale factors for input files"""
    input_files = re.split(', |,', args.input)
    scale_factors = re.split(', |,', str(args.scale_factor))
    scale_factors = list(map(float, scale_factors))

    if len(input_files) != len(scale_factors):
        log.error(f'Incorrect command line for multiple inputs: {len(scale_factors)} scale factors provided for '
                  f'{len(input_files)} input files.')
        sys.exit(-7)

    for i, scale_factor in enumerate(scale_factors):
        if float(scale_factor) < 0:
            log.error(f'Scale factor for input #{i} (counting from zero) is out of range (must be positive).')
            sys.exit(-8)

    return scale_factors


def set_scale_factors(plugin_config: dict, scale_factors: list):
    """Set a scale factor provided for each input"""
    for i, scale_factor in enumerate(scale_factors):
        log.info(f'For input {i} using scale factor of {scale_factor:.7f}')
        plugin_config[f'GNA_SCALE_FACTOR_{i}'] = str(scale_factor)


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = parse_args()

# ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
    log.info('Creating Inference Engine')
    ie = IECore()

# ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation---------------
    if args.model:
        log.info(f'Reading the network: {args.model}')
        # .xml and .bin files
        net = ie.read_network(model=args.model)

# ---------------------------Step 3. Configure input & output----------------------------------------------------------
        log.info('Configuring input and output blobs')
        # Mark layers from args.output_layers as outputs
        if args.output_layers:
            net.add_outputs(get_output_layer_list(net, args, with_ports=True))

        # Get names of input and output blobs
        input_blobs = get_input_layer_list(net, args)
        output_blobs = get_output_layer_list(net, args, with_ports=False)

        # Set input and output precision manually
        for blob_name in input_blobs:
            net.input_info[blob_name].precision = 'FP32'

        for blob_name in output_blobs:
            net.outputs[blob_name].precision = 'FP32'

        net.batch_size = args.batch_size

# ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    devices = args.device.replace('HETERO:', '').split(',')
    plugin_config = {}

    if 'GNA' in args.device:
        gna_device_mode = devices[0] if '_' in devices[0] else 'GNA_AUTO'
        devices[0] = 'GNA'

        plugin_config['GNA_DEVICE_MODE'] = gna_device_mode
        plugin_config['GNA_PRECISION'] = f'I{args.quantization_bits}'

        # Set a GNA scale factor
        if args.import_gna_model:
            if args.scale_factor:
                log.warning(f'Custom scale factor will be used for imported GNA model: {args.import_gna_model}')
                set_scale_factors(plugin_config, parse_scale_factors(args))
            else:
                log.info(f'Using scale factor from the imported GNA model: {args.import_gna_model}')
        else:
            if args.scale_factor:
                set_scale_factors(plugin_config, parse_scale_factors(args))
            else:
                scale_factors = []

                for file_name in re.split(', |,', args.input):
                    first_utterance = next(iter(read_utterance_file(file_name).values()))
                    scale_factors.append(get_scale_factor(first_utterance))

                set_scale_factors(plugin_config, scale_factors)

        if args.export_embedded_gna_model:
            plugin_config['GNA_FIRMWARE_MODEL_IMAGE'] = args.export_embedded_gna_model
            plugin_config['GNA_FIRMWARE_MODEL_IMAGE_GENERATION'] = args.embedded_gna_configuration

        if args.performance_counter:
            plugin_config['PERF_COUNT'] = 'YES'

    device_str = f'HETERO:{",".join(devices)}' if 'HETERO' in args.device else devices[0]

    log.info('Loading the model to the plugin')
    if args.model:
        exec_net = ie.load_network(net, device_str, plugin_config)
    else:
        exec_net = ie.import_network(args.import_gna_model, device_str, plugin_config)
        input_blobs = get_input_layer_list(exec_net, args)
        output_blobs = get_output_layer_list(exec_net, args, with_ports=False)

    if args.input:
        input_files = re.split(', |,', args.input)

        if len(input_blobs) != len(input_files):
            log.error(f'Number of network inputs ({len(input_blobs)}) is not equal '
                      f'to number of ark files ({len(input_files)})')
            sys.exit(-3)

    if args.reference:
        reference_files = re.split(', |,', args.reference)

        if len(output_blobs) != len(reference_files):
            log.error('The number of reference files is not equal to the number of network outputs.')
            sys.exit(-5)

    if args.output:
        output_files = re.split(', |,', args.output)

        if len(output_blobs) != len(output_files):
            log.error('The number of output files is not equal to the number of network outputs.')
            sys.exit(-6)

    if args.export_gna_model:
        log.info(f'Writing GNA Model to {args.export_gna_model}')
        exec_net.export(args.export_gna_model)
        return 0

    if args.export_embedded_gna_model:
        log.info(f'Exported GNA embedded model to file {args.export_embedded_gna_model}')
        log.info(f'GNA embedded model export done for GNA generation {args.embedded_gna_configuration}')
        return 0

# ---------------------------Step 5. Create infer request--------------------------------------------------------------
# load_network() method of the IECore class with a specified number of requests (default 1) returns an ExecutableNetwork
# instance which stores infer requests. So you already created Infer requests in the previous step.

# ---------------------------Step 6. Prepare input---------------------------------------------------------------------
    file_data = [read_utterance_file(file_name) for file_name in input_files]
    input_data = {
        utterance_name: {
            input_blobs[i]: file_data[i][utterance_name] for i in range(len(input_blobs))
        }
        for utterance_name in file_data[0].keys()
    }

    if args.reference:
        references = {output_blobs[i]: read_utterance_file(reference_files[i]) for i in range(len(output_blobs))}

# ---------------------------Step 7. Do inference----------------------------------------------------------------------
    log.info('Starting inference in synchronous mode')
    results = {blob_name: {} for blob_name in output_blobs}
    infer_times = []
    perf_counters = []

    for key in sorted(input_data):
        start_infer_time = default_timer()

        # Reset states between utterance inferences to remove a memory impact
        for request in exec_net.requests:
            for state in request.query_state():
                state.reset()

        result = infer_data(input_data[key], exec_net, input_blobs, output_blobs)

        for blob_name in result.keys():
            results[blob_name][key] = result[blob_name]

        infer_times.append(default_timer() - start_infer_time)
        perf_counters.append(exec_net.requests[0].get_perf_counts())

# ---------------------------Step 8. Process output--------------------------------------------------------------------
    for blob_name in output_blobs:
        for i, key in enumerate(sorted(results[blob_name])):
            log.info(f'Utterance {i} ({key})')
            log.info(f'Output blob name: {blob_name}')
            log.info(f'Frames in utterance: {results[blob_name][key].shape[0]}')
            log.info(f'Total time in Infer (HW and SW): {infer_times[i] * 1000:.2f}ms')

            if args.reference:
                compare_with_reference(results[blob_name][key], references[blob_name][key])

            if args.performance_counter:
                if 'GNA' in args.device:
                    pc = perf_counters[i]
                    total_cycles = int(pc['1.1 Total scoring time in HW']['real_time'])
                    stall_cycles = int(pc['1.2 Stall scoring time in HW']['real_time'])
                    active_cycles = total_cycles - stall_cycles
                    frequency = 10**6
                    if args.arch == 'CORE':
                        frequency *= GNA_CORE_FREQUENCY
                    else:
                        frequency *= GNA_ATOM_FREQUENCY
                    total_inference_time = total_cycles / frequency
                    active_time = active_cycles / frequency
                    stall_time = stall_cycles / frequency
                    log.info('')
                    log.info('Performance Statistics of GNA Hardware')
                    log.info(f'   Total Inference Time: {(total_inference_time * 1000):.4f} ms')
                    log.info(f'   Active Time: {(active_time * 1000):.4f} ms')
                    log.info(f'   Stall Time:  {(stall_time * 1000):.4f} ms')

            log.info('')

    log.info(f'Total sample time: {sum(infer_times) * 1000:.2f}ms')

    if args.output:
        for i, blob_name in enumerate(results):
            write_utterance_file(output_files[i], results[blob_name])
            log.info(f'File {output_files[i]} was created!')

# ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, '
             'for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
