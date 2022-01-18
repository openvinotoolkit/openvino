#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import sys
from timeit import default_timer
from typing import Dict

import numpy as np
from openvino.preprocess import PrePostProcessor
from openvino.runtime import Core, InferRequest, Layout, Type, set_batch

from arg_parser import parse_args
from file_options import read_utterance_file, write_utterance_file
from utils import (GNA_ATOM_FREQUENCY, GNA_CORE_FREQUENCY,
                   compare_with_reference, get_scale_factor, log,
                   parse_outputs_from_args, parse_scale_factors,
                   set_scale_factors)


def do_inference(data: Dict[str, np.ndarray], infer_request: InferRequest, cw_l: int = 0, cw_r: int = 0) -> np.ndarray:
    """Do a synchronous matrix inference"""
    frames_to_infer = {}
    result = {}

    batch_size = infer_request.get_input_tensor(0).shape[0]
    num_of_frames = next(iter(data.values())).shape[0]

    for output in infer_request.model_outputs:
        result[output.any_name] = np.ndarray((num_of_frames, np.prod(tuple(output.shape)[1:])))

    for i in range(-cw_l, num_of_frames + cw_r, batch_size):
        if i < 0:
            index = 0
        elif i >= num_of_frames:
            index = num_of_frames - 1
        else:
            index = i

        for _input in infer_request.model_inputs:
            frames_to_infer[_input.any_name] = data[_input.any_name][index:index + batch_size]
            num_of_frames_to_infer = len(frames_to_infer[_input.any_name])

            # Add [batch_size - num_of_frames_to_infer] zero rows to 2d numpy array
            # Used to infer fewer frames than the batch size
            frames_to_infer[_input.any_name] = np.pad(
                frames_to_infer[_input.any_name],
                [(0, batch_size - num_of_frames_to_infer), (0, 0)],
            )

            frames_to_infer[_input.any_name] = frames_to_infer[_input.any_name].reshape(_input.tensor.shape)

        frame_results = infer_request.infer(frames_to_infer)

        if i - cw_r < 0:
            continue

        for output in frame_results.keys():
            vector_result = frame_results[output].reshape((batch_size, result[output.any_name].shape[1]))
            result[output.any_name][i - cw_r:i - cw_r + batch_size] = vector_result[:num_of_frames_to_infer]

    return result


def main():
    args = parse_args()

# --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')
    core = Core()

# --------------------------- Step 2. Read a model --------------------------------------------------------------------
    if args.model:
        log.info(f'Reading the model: {args.model}')
        # (.xml and .bin files) or (.onnx file)
        model = core.read_model(args.model)

# --------------------------- Step 3. Apply preprocessing -------------------------------------------------------------
        if args.output_layers:
            output_layer_names, output_layer_ports = parse_outputs_from_args(args)
            model.add_outputs(list(zip(output_layer_names, output_layer_ports)))

        ppp = PrePostProcessor(model)

        for i in range(len(model.inputs)):
            ppp.input(i).tensor() \
                .set_element_type(Type.f32) \
                .set_layout(Layout('NC'))  # noqa: N400

            ppp.input(i).model().set_layout(Layout('NC'))

        for i in range(len(model.outputs)):
            ppp.output(i).tensor().set_element_type(Type.f32)

        model = ppp.build()

        if args.context_window_left == args.context_window_right == 0:
            set_batch(model, args.batch_size)
        else:
            set_batch(model, 1)

# ---------------------------Step 4. Configure plugin ---------------------------------------------------------
    devices = args.device.replace('HETERO:', '').split(',')
    plugin_config = {}

    if 'GNA' in args.device:
        gna_device_mode = devices[0] if '_' in devices[0] else 'GNA_AUTO'
        devices[0] = 'GNA'

        plugin_config['GNA_DEVICE_MODE'] = gna_device_mode
        plugin_config['GNA_PRECISION'] = f'I{args.quantization_bits}'
        plugin_config['GNA_EXEC_TARGET'] = args.exec_target
        plugin_config['GNA_PWL_MAX_ERROR_PERCENT'] = str(args.pwl_me)

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
                    _, utterances = read_utterance_file(file_name)
                    scale_factors.append(get_scale_factor(utterances[0]))

                log.info('Using scale factor(s) calculated from first utterance')
                set_scale_factors(plugin_config, scale_factors)

        if args.export_embedded_gna_model:
            plugin_config['GNA_FIRMWARE_MODEL_IMAGE'] = args.export_embedded_gna_model
            plugin_config['GNA_FIRMWARE_MODEL_IMAGE_GENERATION'] = args.embedded_gna_configuration

        if args.performance_counter:
            plugin_config['PERF_COUNT'] = 'YES'

    device_str = f'HETERO:{",".join(devices)}' if 'HETERO' in args.device else devices[0]

# --------------------------- Step 5. Loading model to the device -----------------------------------------------------
    log.info('Loading the model to the plugin')
    if args.model:
        compiled_model = core.compile_model(model, device_str, plugin_config)
    else:
        compiled_model = core.import_model(args.import_gna_model, device_str, plugin_config)

# --------------------------- Exporting GNA model using InferenceEngine AOT API ---------------------------------------
    if args.export_gna_model:
        log.info(f'Writing GNA Model to {args.export_gna_model}')
        compiled_model.export_model(args.export_gna_model)
        return 0

    if args.export_embedded_gna_model:
        log.info(f'Exported GNA embedded model to file {args.export_embedded_gna_model}')
        log.info(f'GNA embedded model export done for GNA generation {args.embedded_gna_configuration}')
        return 0

# --------------------------- Step 6. Set up input --------------------------------------------------------------------
    if args.input_layers:
        input_layer_names = re.split(', |,', args.input_layers)
    else:
        input_layer_names = [_input.any_name for _input in compiled_model.inputs]

    input_file_names = re.split(', |,', args.input)

    if len(input_layer_names) != len(input_file_names):
        log.error(f'Number of network inputs ({len(compiled_model.inputs)}) is not equal '
                  f'to number of ark files ({len(input_file_names)})')
        sys.exit(-3)

    input_file_data = [read_utterance_file(file_name) for file_name in input_file_names]

    infer_data = [
        {
            input_layer_names[j]: input_file_data[j].utterances[i]
            for j in range(len(input_layer_names))
        }
        for i in range(len(input_file_data[0].utterances))
    ]

    if args.output_layers:
        output_layer_names, output_layer_ports = parse_outputs_from_args(args)
        # If a name of output layer contains a port number then concatenate output_layer_names and output_layer_ports
        if ':' in compiled_model.outputs[0].any_name:
            output_layer_names = [f'{output_layer_names[i]}:{output_layer_ports[i]}' for i in range(len(output_layer_names))]
    else:
        output_layer_names = [compiled_model.outputs[0].any_name]

    if args.output:
        output_file_names = re.split(', |,', args.output)

        if len(output_layer_names) != len(output_file_names):
            log.error('The number of output files is not equal to the number of network outputs.')
            sys.exit(-6)

    if args.reference:
        reference_file_names = re.split(', |,', args.reference)

        if len(output_layer_names) != len(reference_file_names):
            log.error('The number of reference files is not equal to the number of network outputs.')
            sys.exit(-5)

        reference_file_data = [read_utterance_file(file_name) for file_name in reference_file_names]

        references = [
            {
                output_layer_names[j]: reference_file_data[j].utterances[i]
                for j in range(len(output_layer_names))
            }
            for i in range(len(input_file_data[0].utterances))
        ]

# --------------------------- Step 7. Create infer request ------------------------------------------------------------
    infer_request = compiled_model.create_infer_request()

# --------------------------- Step 8. Do inference --------------------------------------------------------------------
    log.info('Starting inference in synchronous mode')
    results = []
    total_infer_time = 0

    for i in range(len(infer_data)):
        start_infer_time = default_timer()

        # Reset states between utterance inferences to remove a memory impact
        for state in infer_request.query_state():
            state.reset()

        results.append(do_inference(
            infer_data[i],
            infer_request,
            args.context_window_left,
            args.context_window_right,
        ))

        infer_time = default_timer() - start_infer_time
        total_infer_time += infer_time
        num_of_frames = infer_data[i][input_layer_names[0]].shape[0]
        avg_infer_time_per_frame = infer_time / num_of_frames

# --------------------------- Step 9. Process output ------------------------------------------------------------------
        log.info('')
        log.info(f'Utterance {i}:')
        log.info(f'Total time in Infer (HW and SW): {infer_time * 1000:.2f}ms')
        log.info(f'Frames in utterance: {num_of_frames}')
        log.info(f'Average Infer time per frame: {avg_infer_time_per_frame * 1000:.2f}ms')

        for name in output_layer_names:
            log.info('')
            log.info(f'Output blob name: {name}')
            log.info(f'Number scores per frame: {results[i][name].shape[1]}')

            if args.reference:
                log.info('')
                compare_with_reference(results[i][name], references[i][name])

        if args.performance_counter:
            if 'GNA' in args.device:
                total_cycles = infer_request.profiling_info[0].real_time.total_seconds()
                stall_cycles = infer_request.profiling_info[1].real_time.total_seconds()
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
    log.info(f'Total sample time: {total_infer_time * 1000:.2f}ms')

    if args.output:
        for i, name in enumerate(output_layer_names):
            data = [results[i][name] for i in range(len(input_file_data[0].utterances))]
            write_utterance_file(output_file_names[i], input_file_data[0].keys, data)
            log.info(f'File {output_file_names[i]} was created!')

# ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, '
             'for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
