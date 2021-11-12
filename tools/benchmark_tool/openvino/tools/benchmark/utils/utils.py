# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Core, Function, PartialShape, Layout
from openvino.impl import Type
from openvino.impl.preprocess import PrePostProcessor, InputInfo as InputInfop, OutputInfo, InputTensorInfo, OutputTensorInfo

from .constants import DEVICE_DURATION_IN_SECS, UNKNOWN_DEVICE_TYPE, \
    CPU_DEVICE_NAME, GPU_DEVICE_NAME
from .logging import logger

import json
import re
import numpy as np

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
        8: "Querying optimal runtime parameters",
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


def get_element_type(precision):
    format_map = {
      'FP32' : Type.f32,
      'I32'  : Type.i32,
      'I64'  : Type.i64,
      'FP16' : Type.f16,
      'I16'  : Type.i16,
      'U16'  : Type.u16,
      'I8'   : Type.i8,
      'U8'   : Type.u8,
      'BOOL' : Type.boolean,
    }
    if precision in format_map.keys():
        return format_map[precision]
    raise Exception("Can't find openvino element type for precision: " + precision)


def process_precision(function: Function, app_inputs_info, input_precision: str, output_precision: str, input_output_precision: str):
    pre_post_processor = PrePostProcessor()
    if input_precision:
        element_type = get_element_type(input_precision)
        for i in range(function.inputs):
            pre_post_processor.input(InputInfop(i).tensor(InputTensorInfo().set_element_type(element_type)))
    if output_precision:
        element_type = get_element_type(output_precision)
        for i in range(function.outputs):
            pre_post_processor.output(OutputInfo(i).tensor(OutputTensorInfo().set_element_type(element_type)))
    if input_output_precision:
        user_precision_map = _parse_arg_map(input_output_precision)
        input_names = get_input_output_names(function.inputs)
        output_names = get_input_output_names(function.outputs)
        for node_name, precision in user_precision_map.items():
            user_precision_map[node_name] = get_element_type(precision)
        for name, element_type in user_precision_map.items():
            if name in input_names:
                pre_post_processor.input(InputInfop(name).tensor(InputTensorInfo().set_element_type(element_type)))
            elif name in output_names:
                pre_post_processor.output(OutputInfo(i).tensor(OutputTensorInfo().set_element_type(element_type)))
            else:
                raise Exception(f"Node '{name}' does not exist in network")
    function = pre_post_processor.build(function)
    # update app_inputs_info
    for i in range(function.inputs):
        app_inputs_info[i].element_type = function.input(i).get_element_type()


def _parse_arg_map(arg_map: str):
    arg_map = arg_map.replace(" ", "")
    pairs = [x.strip() for x in arg_map.split(',')]

    parsed_map = {}
    for pair in pairs:
        key_value = [x.strip() for x in pair.split(':')]
        parsed_map.update({key_value[0]:key_value[1]})

    return parsed_map


"""def print_inputs_and_outputs_info(function: Function):
    for input in function.inputs:
        logger.info(f"Network input '{input.get_node().name}', type {input.get_element_type()}, "
                                                    f"s ")
    output_info = ie_network.outputs
    for key in output_info.keys():
        info = output_info[key]
        logger.info(f"Network output '{key}' precision {info.precision}, "
                                        f"dimensions ({info.layout}): "
                                        f"{' '.join(str(x) for x in info.shape)}")"""


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
    return [d for d in devices.split(',')]


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


def process_help_inference_string(benchmark_app, device_number_streams):
    output_string = f'Start inference {benchmark_app.api_type}hronously'
    if benchmark_app.api_type == 'async':
        output_string += f', {benchmark_app.nireq} inference requests'

        device_ss = ''
        for device, streams in device_number_streams.items():
            device_ss += ', ' if device_ss else ''
            device_ss += f'{streams} streams for {device}'

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

def get_input_output_names(output_nodes):
    return [output_node.get_node().get_name() for output_node in output_nodes]

def parse_input_parameters(parameter_string, input_names):
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
                    return_value  = { k:value for k in input_names }
                    break
        else:
            raise Exception(f"Can't parse input parameter: {parameter_string}")
    return return_value

def parse_scale_or_mean(parameter_string, input_info):
    # Parse parameter string like "input0[value0],input1[value1]" or "[value]" (applied to all inputs)
    return_value = {}
    if parameter_string:
        matches = re.findall(r'(.*?)\[(.*?)\],?', parameter_string)
        if matches:
            for match in matches:
                input_name, value = match
                f_value = np.array(value.split(",")).astype(np.float)
                if input_name != '':
                    return_value[input_name] = f_value
                else:
                    for input in input_info:
                        if input.is_image:
                            return_value[input.tensor_name] = f_value
        else:
            raise Exception(f"Can't parse input parameter: {parameter_string}")
    return return_value

class InputInfo:
    def __init__(self):
        self.element_type = None
        self.layout = Layout()
        self.shape = None
        self.scale = []
        self.mean = []
        self.tensor_name = None

    @property
    def is_image(self):
        if str(self.layout) not in [ "NCHW", "NHWC", "CHW", "HWC" ]:
            return False
        return self.channels == 3

    @property
    def is_image_info(self):
        if str(self.layout) != "NC":
            return False
        return self.channels >= 2

    def getDimentionByLayout(self, character):
        return self.shape[self.layout.get_index_by_name(character)]

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

def get_inputs_info(shape_string, layout_string, batch_size, scale_string, mean_string, inputs, parameters = None):
    input_names = get_input_output_names(inputs)
    shape_map = parse_input_parameters(shape_string, input_names)
    layout_map = parse_input_parameters(layout_string, input_names)
    reshape = False
    input_info = []
    for i in range(len(inputs)):
        info = InputInfo()
        # Precision
        info.element_type = inputs[i].get_element_type()
        # Shape
        input_name = inputs[i].get_node().get_name()
        info.tensor_name = input_name
        if input_name in shape_map.keys():
            parsed_shape = PartialShape(int(dim) for dim in shape_map[input_name].split(','))
            info.shape = parsed_shape
            reshape = True
        else:
            info.shape = inputs[i].get_partial_shape()

        if input_name in layout_map.keys():
            info.layout = Layout(layout_map[input_name])
        elif parameters:
            info.layout = parameters[i].get_layout()

        # Update shape with batch if needed
        if batch_size != 0:
            batch_index = info.layout.get_index_by_name('N') if info.layout.has_name('N') else -1
            if batch_index != -1 and info.shape[batch_index] != batch_size:
                info.shape[batch_index] = batch_size
                reshape = True
        input_info.append(info)

    # Update scale, mean
    scale_map = parse_scale_or_mean(scale_string, input_names)
    mean_map = parse_scale_or_mean(mean_string, input_names)

    for input in input_info:
        if input.is_image:
            input.scale = np.ones(3)
            input.mean = np.zeros(3)

            if input.tensor_name in scale_map:
                input.scale = scale_map[input.tensor_name]
            if input.tensor_name in mean_map:
                input.mean = mean_map[input.tensor_name]

    return input_info, reshape

def get_batch_size(inputs_info):
    batch_size = 0
    for _, info in inputs_info.items():
        batch_index = info.layout.get_index_by_name('N') if info.layout.has_name('N') else -1
        if batch_index != -1:
            if batch_size == 0:
                batch_size = info.batch
            elif batch_size != info.shape[batch_index]:
                raise Exception("Can't deterimine batch size: batch is different for different inputs!")
    if batch_size == 0:
        batch_size = 1
    return batch_size

def show_available_devices():
    print("\nAvailable target devices:  ", ("  ".join(Core().available_devices)))

def dump_config(filename, config):
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(filename, config):
    with open(filename) as f:
        config.update(json.load(f))
