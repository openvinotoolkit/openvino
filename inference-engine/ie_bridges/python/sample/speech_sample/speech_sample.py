#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging as log
import re
import sys
from timeit import default_timer
from typing import Any, IO

import numpy as np
from openvino.inference_engine import ExecutableNetwork, IECore


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    model = parser.add_mutually_exclusive_group(required=True)

    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    model.add_argument('-m', '--model', type=str,
                       help='Path to an .xml file with a trained model (required if -rg is missing).')
    model.add_argument('-rg', '--import_gna_model', type=str,
                       help='Read GNA model from file using path/filename provided (required if -m is missing).')
    args.add_argument('-i', '--input', required=True, type=str, help='Required. Path to an utterance file.')
    args.add_argument('-o', '--output', type=str, help='Optional. Output file name to save inference results.')
    args.add_argument('-r', '--reference', type=str,
                      help='Optional. Read reference score file and compare scores.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify a target device to infer on. '
                      'CPU, GPU, MYRIAD, GNA_AUTO, GNA_HW, GNA_SW_FP32, GNA_SW_EXACT and HETERO with combination of GNA'
                      ' as the primary device and CPU as a secondary (e.g. HETERO:GNA,CPU) are supported. '
                      'The sample will look for a suitable plugin for device specified. Default value is CPU.')
    args.add_argument('-bs', '--batch_size', default=1, type=int, help='Optional. Batch size 1-8 (default 1).')
    args.add_argument('-qb', '--quantization_bits', default=16, type=int,
                      help='Optional. Weight bits for quantization: 8 or 16 (default 16).')
    args.add_argument('-wg', '--export_gna_model', type=str,
                      help='Optional. Write GNA model to file using path/filename provided.')
    # TODO: Find a model that applicable for -we option
    args.add_argument('-we', '--export_embedded_gna_model', type=str, help=argparse.SUPPRESS)
    # TODO: Find a model that applicable for -we_gen option
    args.add_argument('-we_gen', '--embedded_gna_configuration', default='GNA1', type=str, help=argparse.SUPPRESS)
    args.add_argument('-iname', '--input_layers', type=str,
                      help='Optional. Layer names for input blobs. The names are separated with ",". '
                      'Allows to change the order of input layers for -i flag. Example: Input1,Input2')
    args.add_argument('-oname', '--output_layers', type=str,
                      help='Optional. Layer names for output blobs. The names are separated with ",". '
                      'Allows to change the order of output layers for -o flag. Example: Output1:port,Output2:port.')

    return parser.parse_args()


def get_scale_factor(matrix: np.ndarray) -> float:
    """Get scale factor for quantization using utterance matrix"""
    # Max to find scale factor
    target_max = 16384
    max_val = np.max(matrix)
    if max_val == 0:
        return 1.0
    else:
        return target_max / max_val


def read_ark_file(file_name: str) -> dict:
    """Read utterance matrices from a .ark file"""
    def read_key(input_file: IO[Any]) -> str:
        """Read a identifier of utterance matrix"""
        key = ''
        while True:
            char = input_file.read(1).decode()
            if char in ('', ' '):
                break
            else:
                key += char

        return key

    def read_matrix(input_file: IO[Any]) -> np.ndarray:
        """Read a utterance matrix"""
        header = input_file.read(5).decode()
        if 'FM' in header:
            num_of_bytes = 4
            dtype = 'float32'
        elif 'DM' in header:
            num_of_bytes = 8
            dtype = 'float64'

        _, rows, _, cols = np.frombuffer(input_file.read(10), 'int8, int32, int8, int32')[0]
        buffer = input_file.read(rows * cols * num_of_bytes)
        vector = np.frombuffer(buffer, dtype)
        matrix = np.reshape(vector, (rows, cols))

        return matrix

    utterances = {}
    with open(file_name, 'rb') as input_file:
        while True:
            key = read_key(input_file)
            if not key:
                break
            utterances[key] = read_matrix(input_file)

    return utterances


def write_ark_file(file_name: str, utterances: dict):
    """Write utterance matrices to a .ark file"""
    with open(file_name, 'wb') as output_file:
        for key, matrix in sorted(utterances.items()):
            # write a matrix key
            output_file.write(key.encode())
            output_file.write(' '.encode())
            output_file.write('\0B'.encode())

            # write a matrix precision
            if matrix.dtype == 'float32':
                output_file.write('FM '.encode())
            elif matrix.dtype == 'float64':
                output_file.write('DM '.encode())

            # write a matrix shape
            output_file.write('\04'.encode())
            output_file.write(matrix.shape[0].to_bytes(4, byteorder='little', signed=False))
            output_file.write('\04'.encode())
            output_file.write(matrix.shape[1].to_bytes(4, byteorder='little', signed=False))

            # write a matrix data
            output_file.write(matrix.tobytes())


def infer_data(data: dict, exec_net: ExecutableNetwork, input_blobs: list, output_blobs: list) -> np.ndarray:
    """Do a synchronous matrix inference"""
    matrix_shape = next(iter(data.values())).shape
    result = {}

    for blob_name in output_blobs:
        batch_size, num_of_dims = exec_net.outputs[blob_name].shape
        result[blob_name] = np.ndarray((matrix_shape[0], num_of_dims))

    slice_begin = 0
    slice_end = batch_size

    while True:
        vectors = {blob_name: data[blob_name][slice_begin:slice_end] for blob_name in input_blobs}
        vector_shape = next(iter(vectors.values())).shape

        if vector_shape[0] < batch_size:
            for i in range(vector_shape[0]):
                input_data = {key: value[i] for key, value in vectors.items()}
                vector_results = exec_net.infer(input_data)

                for blob_name in output_blobs:
                    result[blob_name][slice_begin] = vector_results[blob_name][0]

                slice_begin += 1
        else:
            vector_results = exec_net.infer(vectors)

            for blob_name in output_blobs:
                result[blob_name][slice_begin:slice_end] = vector_results[blob_name]

            slice_begin += batch_size
            slice_end += batch_size

        if slice_begin >= matrix_shape[0]:
            return result


def compare_with_reference(result: np.ndarray, reference: np.ndarray):
    error_matrix = np.absolute(result - reference)

    max_error = np.max(error_matrix)
    sum_error = np.sum(error_matrix)
    avg_error = sum_error / error_matrix.size
    sum_square_error = np.sum(np.square(error_matrix))
    avg_rms_error = np.sqrt(sum_square_error / error_matrix.size)
    stdev_error = np.sqrt(sum_square_error / error_matrix.size - avg_error * avg_error)

    log.info(f'max error: {max_error}')
    log.info(f'avg error: {avg_error}')
    log.info(f'avg rms error: {avg_rms_error}')
    log.info(f'stdev error: {stdev_error}')


def read_utterance_file(file_name: str) -> dict:
    """Read utterance matrices from a file"""
    file_extension = file_name.split('.')[-1]

    if file_extension == 'ark':
        return read_ark_file(file_name)
    elif file_extension == 'npz':
        return dict(np.load(file_name))
    else:
        log.error(f'The file {file_name} cannot be read. The sample supports only .ark and .npz files.')
        sys.exit(-1)


def write_utterance_file(file_name: str, utterances: dict):
    """Write utterance matrices to a file"""
    file_extension = file_name.split('.')[-1]

    if file_extension == 'ark':
        write_ark_file(file_name, utterances)
    elif file_extension == 'npz':
        np.savez(file_name, **utterances)
    else:
        log.error(f'The file {file_name} cannot be written. The sample supports only .ark and .npz files.')
        sys.exit(-2)


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = parse_args()

# ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
    log.info('Creating Inference Engine')
    ie = IECore()

# ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
    if args.model:
        log.info(f'Reading the network: {args.model}')
        # (.xml and .bin files) or (.onnx file)
        net = ie.read_network(model=args.model)

# ---------------------------Step 3. Configure input & output----------------------------------------------------------
        log.info('Configuring input and output blobs')
        # Get names of input and output blobs
        if args.input_layers:
            input_blobs = re.split(', |,', args.input_layers)
        else:
            input_blobs = [next(iter(net.input_info))]

        if args.output_layers:
            output_name_port = [output.split(':') for output in re.split(', |,', args.output_layers)]
            try:
                output_name_port = [(blob_name, int(port)) for blob_name, port in output_name_port]
            except ValueError:
                log.error('Output Parameter does not have a port.')
                sys.exit(-4)

            net.add_outputs(output_name_port)

            output_blobs = [blob_name for blob_name, port in output_name_port]
        else:
            output_blobs = [list(net.outputs.keys())[-1]]

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

        # Get a GNA scale factor
        if args.import_gna_model:
            log.info(f'Using scale factor from the imported GNA model: {args.import_gna_model}')
        else:
            utterances = read_utterance_file(args.input.split(',')[0])
            key = sorted(utterances)[0]
            scale_factor = get_scale_factor(utterances[key])
            log.info(f'Using scale factor of {scale_factor} calculated from first utterance.')

            plugin_config['GNA_SCALE_FACTOR'] = str(scale_factor)

        if args.export_embedded_gna_model:
            plugin_config['GNA_FIRMWARE_MODEL_IMAGE'] = args.export_embedded_gna_model
            plugin_config['GNA_FIRMWARE_MODEL_IMAGE_GENERATION'] = args.embedded_gna_configuration

    device_str = f'HETERO:{",".join(devices)}' if 'HETERO' in args.device else devices[0]

    log.info('Loading the model to the plugin')
    if args.model:
        exec_net = ie.load_network(net, device_str, plugin_config)
    else:
        exec_net = ie.import_network(args.import_gna_model, device_str, plugin_config)
        input_blobs = [next(iter(exec_net.input_info))]
        output_blobs = [list(exec_net.outputs.keys())[-1]]

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

    # TODO: Find a model that applicable for -we and -we_gen options
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

    for key in sorted(input_data):
        start_infer_time = default_timer()
        result = infer_data(input_data[key], exec_net, input_blobs, output_blobs)

        for blob_name in result.keys():
            results[blob_name][key] = result[blob_name]

        infer_times.append(default_timer() - start_infer_time)

# ---------------------------Step 8. Process output--------------------------------------------------------------------
    for blob_name in output_blobs:
        for i, key in enumerate(sorted(results[blob_name])):
            log.info(f'Utterance {i} ({key})')
            log.info(f'Output blob name: {blob_name}')
            log.info(f'Frames in utterance: {results[blob_name][key].shape[0]}')
            log.info(f'Total time in Infer (HW and SW): {infer_times[i] * 1000:.2f}ms')

            if args.reference:
                compare_with_reference(results[blob_name][key], references[blob_name][key])

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
