# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.inference_engine import IENetwork,IECore

from .constants import DEVICE_DURATION_IN_SECS, UNKNOWN_DEVICE_TYPE, \
    CPU_DEVICE_NAME, GPU_DEVICE_NAME
from .logging import logger

import json
import re

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@static_vars(step_id=0)
def next_step(additional_info='', step_id=0):
    step_names = {
        1: "Parsing and validating input arguments",
        2: "Loading Inference Engine",
        3: "Setting device configuration",
        4: "Reading network files",
        5: "Resizing network to match image sizes and given batch",
        6: "Configuring input of the model",
        7: "Loading the model to the device",
        8: "Setting optimal runtime parameters",
        9: "Creating infer requests and filling input blobs with images",
        10: "Measuring performance",
        11: "Dumping statistics report",
    }
    if step_id != 0:
        next_step.step_id = step_id
    else:
        next_step.step_id += 1

    if next_step.step_id not in step_names.keys():
        raise Exception(f'Step ID {next_step.step_id} is out of total steps number {str(len(step_names))}')

    step_info_template = '[Step {}/{}] {}'
    step_name = step_names[next_step.step_id] + (f' ({additional_info})' if additional_info else '')
    step_info_template = step_info_template.format(next_step.step_id, len(step_names), step_name)
    print(step_info_template)

def process_precision(ie_network: IENetwork, app_inputs_info, input_precision: str, output_precision: str, input_output_precision: str):
    if input_precision:
        _configure_network_inputs(ie_network, app_inputs_info, input_precision)
    if output_precision:
        _configure_network_outputs(ie_network, output_precision)
    if input_output_precision:
        _configure_network_inputs_and_outputs(ie_network, input_output_precision)
    input_info = ie_network.input_info
    for key in app_inputs_info.keys():
        ## if precision for input set by user, then set it to app_inputs
        ## if it an image, set U8
        if input_precision or (input_output_precision and key in input_output_precision.keys()):
            app_inputs_info[key].precision = input_info[key].precision
        elif app_inputs_info[key].is_image:
            app_inputs_info[key].precision = 'U8'
            input_info[key].precision = 'U8'

def _configure_network_inputs(ie_network: IENetwork, app_inputs_info, input_precision: str):
    input_info = ie_network.input_info

    for key in input_info.keys():
        app_inputs_info[key].precision = input_precision
        input_info[key].precision = input_precision

def _configure_network_outputs(ie_network: IENetwork, output_precision: str):
    output_info = ie_network.outputs

    for key in output_info.keys():
        output_info[key].precision = output_precision

def _configure_network_inputs_and_outputs(ie_network: IENetwork, input_output_precision: str):
    if not input_output_precision:
        raise Exception("Input/output precision is empty")

    user_precision_map = _parse_arg_map(input_output_precision)

    input_info = ie_network.input_info
    output_info = ie_network.outputs

    for key, value in user_precision_map.items():
        if key in input_info:
            input_info[key].precision = value
        elif key in output_info:
            output_info[key].precision = value
        else:
            raise Exception(f"Element '{key}' does not exist in network")

def _parse_arg_map(arg_map: str):
    arg_map = arg_map.replace(" ", "")
    pairs = [x.strip() for x in arg_map.split(',')]

    parsed_map = {}
    for pair in pairs:
        key_value = [x.strip() for x in pair.split(':')]
        parsed_map.update({key_value[0]:key_value[1]})

    return parsed_map

def print_inputs_and_outputs_info(ie_network: IENetwork):
    input_info = ie_network.input_info
    for key in input_info.keys():
        tensor_desc = input_info[key].tensor_desc
        logger.info(f"Network input '{key}' precision {tensor_desc.precision}, "
                                                    f"dimensions ({tensor_desc.layout}): "
                                                    f"{' '.join(str(x) for x in tensor_desc.dims)}")
    output_info = ie_network.outputs
    for key in output_info.keys():
        info = output_info[key]
        logger.info(f"Network output '{key}' precision {info.precision}, "
                                        f"dimensions ({info.layout}): "
                                        f"{' '.join(str(x) for x in info.shape)}")

def get_number_iterations(number_iterations: int, nireq: int, api_type: str):
    niter = number_iterations

    if api_type == 'async' and niter:
        niter = int((niter + nireq - 1) / nireq) * nireq
        if number_iterations != niter:
            logger.warning('Number of iterations was aligned by request number '
                           f'from {number_iterations} to {niter} using number of requests {nireq}')

    return niter


def get_duration_seconds(time, number_iterations, device):
    if time:
        # time limit
        return time

    if not number_iterations:
        return get_duration_in_secs(device)
    return 0


def get_duration_in_milliseconds(duration):
    return duration * 1000


def get_duration_in_secs(target_device):
    duration = 0
    for device in DEVICE_DURATION_IN_SECS:
        if device in target_device:
            duration = max(duration, DEVICE_DURATION_IN_SECS[device])

    if duration == 0:
        duration = DEVICE_DURATION_IN_SECS[UNKNOWN_DEVICE_TYPE]
        logger.warning(f'Default duration {duration} seconds is used for unknown device {target_device}')

    return duration


def parse_devices(device_string):
    if device_string in ['MULTI', 'HETERO']:
        return list()
    devices = device_string
    if ':' in devices:
        devices = devices.partition(':')[2]
    return [d[:d.index('(')] if '(' in d else
            d[:d.index('.')] if '.' in d else d for d in devices.split(',')]


def parse_nstreams_value_per_device(devices, values_string):
    # Format: <device1>:<value1>,<device2>:<value2> or just <value>
    result = {}
    if not values_string:
        return result
    device_value_strings = values_string.split(',')
    for device_value_string in device_value_strings:
        device_value_vec = device_value_string.split(':')
        if len(device_value_vec) == 2:
            device_name = device_value_vec[0]
            nstreams = device_value_vec[1]
            if device_name in devices:
                result[device_name] = nstreams
            else:
                raise Exception("Can't set nstreams value " + str(nstreams) +
                                " for device '" + device_name + "'! Incorrect device name!");
        elif len(device_value_vec) == 1:
            nstreams = device_value_vec[0]
            for device in devices:
                result[device] = nstreams
        elif not device_value_vec:
            raise Exception('Unknown string format: ' + values_string)
    return result


def process_help_inference_string(benchmark_app):
    output_string = f'Start inference {benchmark_app.api_type}hronously'
    if benchmark_app.api_type == 'async':
        output_string += f', {benchmark_app.nireq} inference requests'

        device_ss = ''
        if CPU_DEVICE_NAME in benchmark_app.device:
            device_ss += str(benchmark_app.ie.get_config(CPU_DEVICE_NAME, 'CPU_THROUGHPUT_STREAMS'))
            device_ss += f' streams for {CPU_DEVICE_NAME}'
        if GPU_DEVICE_NAME in benchmark_app.device:
            device_ss += ', ' if device_ss else ''
            device_ss += str(benchmark_app.ie.get_config(GPU_DEVICE_NAME, 'GPU_THROUGHPUT_STREAMS'))
            device_ss += f' streams for {GPU_DEVICE_NAME}'

        if device_ss:
            output_string += ' using ' + device_ss

    limits = ''

    if benchmark_app.niter and not benchmark_app.duration_seconds:
        limits += f'{benchmark_app.niter} iterations'

    if benchmark_app.duration_seconds:
        limits += f'{get_duration_in_milliseconds(benchmark_app.duration_seconds)} ms duration'
    if limits:
        output_string += ', limits: ' + limits

    return output_string


def dump_exec_graph(exe_network, exec_graph_path):
    try:
        exec_graph_info = exe_network.get_exec_graph_info()
        exec_graph_info.serialize(exec_graph_path)
        logger.info(f'Executable graph is stored to {exec_graph_path}')
        del exec_graph_info
    except Exception as e:
        logger.exception(e)


def print_perf_counters(perf_counts_list):
    for ni in range(len(perf_counts_list)):
        perf_counts = perf_counts_list[ni]
        total_time = 0
        total_time_cpu = 0
        logger.info(f"Performance counts for {ni}-th infer request")
        for layer, stats in sorted(perf_counts.items(), key=lambda x: x[1]['execution_index']):
            max_layer_name = 30
            print(f"{layer[:max_layer_name - 4] + '...' if (len(layer) >= max_layer_name) else layer:<30}"
                                                                f"{stats['status']:<15}"
                                                                f"{'layerType: ' + str(stats['layer_type']):<30}"
                                                                f"{'realTime: ' + str(stats['real_time']):<20}"
                                                                f"{'cpu: ' + str(stats['cpu_time']):<20}"
                                                                f"{'execType: ' + str(stats['exec_type']):<20}")
            total_time += stats['real_time']
            total_time_cpu += stats['cpu_time']
        print(f'Total time:     {total_time} microseconds')
        print(f'Total CPU time: {total_time_cpu} microseconds\n')

def get_command_line_arguments(argv):
    parameters = []
    arg_name = ''
    arg_value = ''
    for arg in argv[1:]:
        if '=' in arg:
            arg_name, arg_value = arg.split('=')
            parameters.append((arg_name, arg_value))
            arg_name = ''
            arg_value = ''
        else:
          if arg[0] == '-':
              if arg_name is not '':
                parameters.append((arg_name, arg_value))
                arg_value = ''
              arg_name = arg
          else:
              arg_value = arg
    if arg_name is not '':
        parameters.append((arg_name, arg_value))
    return parameters

def parse_input_parameters(parameter_string, input_info):
    # Parse parameter string like "input0[value0],input1[value1]" or "[value]" (applied to all inputs)
    return_value = {}
    if parameter_string:
        matches = re.findall(r'(.*?)\[(.*?)\],?', parameter_string)
        if matches:
            for match in matches:
                input_name, value = match
                if input_name != '':
                    return_value[input_name] = value
                else:
                    return_value  = { k:value for k in input_info.keys() }
                    break
        else:
            raise Exception(f"Can't parse input parameter: {parameter_string}")
    return return_value

class InputInfo:
    def __init__(self):
        self.precision = None
        self.layout = ""
        self.shape = []

    @property
    def is_image(self):
        if self.layout not in [ "NCHW", "NHWC", "CHW", "HWC" ]:
            return False
        return self.channels == 3

    @property
    def is_image_info(self):
        if self.layout != "NC":
            return False
        return self.channels >= 2

    def getDimentionByLayout(self, character):
        if character not in self.layout:
            raise Exception(f"Error: Can't get {character} from layout {self.layout}")
        return self.shape[self.layout.index(character)]

    @property
    def width(self):
        return self.getDimentionByLayout("W")

    @property
    def height(self):
        return self.getDimentionByLayout("H")

    @property
    def channels(self):
        return self.getDimentionByLayout("C")

    @property
    def batch(self):
        return self.getDimentionByLayout("N")

    @property
    def depth(self):
        return self.getDimentionByLayout("D")

def get_inputs_info(shape_string, layout_string, batch_size, input_info):
    shape_map = parse_input_parameters(shape_string, input_info)
    layout_map = parse_input_parameters(layout_string, input_info)
    reshape = False
    info_map = {}
    for name, descriptor in input_info.items():
        info = InputInfo()
        # Precision
        info.precision = descriptor.precision
        # Shape
        if name in shape_map.keys():
            parsed_shape = [int(dim) for dim in shape_map[name].split(',')]
            info.shape = parsed_shape
            reshape = True
        else:
            info.shape = descriptor.input_data.shape
        # Layout
        info.layout = layout_map[name].upper() if name in layout_map.keys() else descriptor.tensor_desc.layout
        # Update shape with batch if needed
        if batch_size != 0:
            batch_index = info.layout.index('N') if 'N' in info.layout else -1
            if batch_index != -1 and info.shape[batch_index] != batch_size:
                info.shape[batch_index] = batch_size
                reshape = True
        info_map[name] = info
    return info_map, reshape

def get_batch_size(inputs_info):
    batch_size = 0
    for _, info in inputs_info.items():
        batch_index = info.layout.index('N') if 'N' in info.layout else -1
        if batch_index != -1:
            if batch_size == 0:
                batch_size = info.shape[batch_index]
            elif batch_size != info.shape[batch_index]:
                raise Exception("Can't deterimine batch size: batch is different for different inputs!")
    if batch_size == 0:
        batch_size = 1
    return batch_size

def show_available_devices():
    ie = IECore()
    print("\nAvailable target devices:  ", ("  ".join(ie.available_devices)))

def dump_config(filename, config):
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(filename, config):
    with open(filename) as f:
        config.update(json.load(f))
