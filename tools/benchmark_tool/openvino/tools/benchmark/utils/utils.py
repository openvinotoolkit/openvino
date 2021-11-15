# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datetime
from openvino import Core, Function, PartialShape, Dimension, Layout
from openvino.impl import Type
from openvino.impl.preprocess import PrePostProcessor, InputInfo, OutputInfo, InputTensorInfo, OutputTensorInfo

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
    # TODO: raise python exception if precion in input_output_precision rewrite input_precision/output_precision
    pre_post_processor = PrePostProcessor()
    if input_precision:
        element_type = get_element_type(input_precision)
        for i in range(len(function.inputs)):
            pre_post_processor.input(InputInfo(i).tensor(InputTensorInfo().set_element_type(element_type)))
            app_inputs_info[i].element_type = element_type
    if output_precision:
        element_type = get_element_type(output_precision)
        for i in range(len(function.outputs)):
            pre_post_processor.output(OutputInfo(i).tensor(OutputTensorInfo().set_element_type(element_type)))
    user_precision_map = {}
    if input_output_precision:
        user_precision_map = _parse_arg_map(input_output_precision)
        input_names = get_input_output_names(function.get_parameters())
        output_names = get_input_output_names(function.get_results())
        for node_name, precision in user_precision_map.items():
            user_precision_map[node_name] = get_element_type(precision)
        for name, element_type in user_precision_map.items():
            if name in input_names:
                port = input_names.index(name)
                pre_post_processor.input(InputInfo(port).tensor(InputTensorInfo().set_element_type(element_type)))
            elif name in output_names:
                port = output_names.index(name)
                pre_post_processor.output(OutputInfo(port).tensor(OutputTensorInfo().set_element_type(element_type)))
            else:
                raise Exception(f"Node '{name}' does not exist in network")
    # update app_inputs_info
    if not input_precision:
        inputs = function.inputs
        for i in range(len(inputs)):
            if app_inputs_info[i].name in user_precision_map.keys():
                app_inputs_info[i].element_type = user_precision_map[app_inputs_info[i].name]
            elif app_inputs_info[i].is_image:
                app_inputs_info[i].element_type = Type.u8
                pre_post_processor.input(InputInfo(i).tensor(InputTensorInfo().set_element_type(Type.u8)))
            else:
                app_inputs_info[i].element_type = inputs[i].get_element_type()
    function = pre_post_processor.build(function)


def _parse_arg_map(arg_map: str):
    arg_map = arg_map.replace(" ", "")
    pairs = [x.strip() for x in arg_map.split(',')]

    parsed_map = {}
    for pair in pairs:
        key_value = [x.strip() for x in pair.split(':')]
        parsed_map.update({key_value[0]:key_value[1]})

    return parsed_map


def get_precision(element_type: Type):
    format_map = {
      'f32' : 'FP32',
      'i32'  : 'I32',
      'i64'  : 'I64',
      'f16' : 'FP16',
      'i16'  : 'I16',
      'u16'  : 'U16',
      'i8'   : 'I8',
      'u8'   : 'U8',
      'boolean' : 'BOOL',
    }
    if element_type.get_type_name() in format_map.keys():
        return format_map[element_type.get_type_name()]
    raise Exception("Can't find  precision for openvino element type: " + str(element_type))

def print_inputs_and_outputs_info(function: Function):
    parameters = function.get_parameters()
    input_names = get_input_output_names(parameters)
    for i in range(len(parameters)):
        logger.info(f"Network input '{input_names[i]}' precision {get_precision(parameters[i].get_element_type())}, "
                                                    f"dimensions ({str(parameters[i].get_layout())}): "
                                                    f"{' '.join(str(x) for x in parameters[i].get_partial_shape())}")
    results = function.get_results()
    output_names = get_input_output_names(results)
    results = function.get_results()
    for i in range(len(results)):
        logger.info(f"Network output '{output_names[i]}' precision {get_precision(results[i].get_element_type())}, "
                                        f"dimensions ({str(results[i].get_layout())}): "
                                        f"{' '.join(str(x) for x in  results[i].get_output_shape(0))}")


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
    pass # TODO: is there any way to do it in new api?
    #try:
        #exec_graph_info = exe_network.get_runtime_function()
        #exec_graph_info.serialize(exec_graph_path)
        #logger.info(f'Executable graph is stored to {exec_graph_path}')
        #del exec_graph_info
    #except Exception as e:
        #logger.exception(e)


def print_perf_counters(perf_counts_list):
    max_layer_name = 30
    for ni in range(len(perf_counts_list)):
        perf_counts = perf_counts_list[ni]
        total_time = datetime.timedelta()
        total_time_cpu = datetime.timedelta()
        logger.info(f"Performance counts for {ni}-th infer request")
        for pi in perf_counts:
            print(f"{pi.node_name[:max_layer_name - 4] + '...' if (len(pi.node_name) >= max_layer_name) else pi.node_name:<30}"
                                                                f"{str(pi.status):<15}"
                                                                f"{'layerType: ' + pi.node_type:<30}"
                                                                f"{'realTime: ' + str(pi.real_time):<20}"
                                                                f"{'cpu: ' +  str(pi.cpu_time):<20}"
                                                                f"{'execType: ' + pi.exec_type:<20}")
            total_time += pi.real_time
            total_time_cpu += pi.cpu_time
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


def get_input_output_names(nodes):
    return [node.get_friendly_name() for node in nodes] # get_friendly_name() or get_name() ?


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
                            return_value[input.name] = f_value
        else:
            raise Exception(f"Can't parse input parameter: {parameter_string}")
    return return_value


class AppInputInfo:
    def __init__(self):
        self.element_type = None
        self.layout = ""
        self.shape = None
        self.scale = []
        self.mean = []
        self.name = None

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
        return len(self.shape[self.layout.index(character)])

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


def get_inputs_info(shape_string, layout_string, batch_size, scale_string, mean_string, parameters):
    input_names = get_input_output_names(parameters)
    shape_map = parse_input_parameters(shape_string, input_names)
    layout_map = parse_input_parameters(layout_string, input_names)
    reshape = False
    input_info = []
    for i in range(len(parameters)):
        info = AppInputInfo()
        # Name
        info.name = input_names[i]
        # Shape
        if info.name in shape_map.keys():
            parsed_shape = PartialShape(list(int(dim) for dim in shape_map[info.name].split(','))) # TODO: update to support dynamic shapes
            info.shape = parsed_shape
            reshape = True
        else:
            info.shape = parameters[i].get_partial_shape()
        # Layout
        if info.name in layout_map.keys():
            info.layout = layout_map[info.name]
        elif parameters[i].get_layout() != Layout():
            info.layout = str(parameters[i].get_layout())
        else:
            info.layout = "NCHW"

        # Update shape with batch if needed
        if batch_size != 0:
            batch_index = info.layout.index('N') if 'N' in info.layout else -1
            if batch_index != -1 and info.shape[batch_index] != Dimension(batch_size):
                info.shape[batch_index] = Dimension(batch_size)
                reshape = True
        input_info.append(info)

    # Update scale, mean
    scale_map = parse_scale_or_mean(scale_string, input_info)
    mean_map = parse_scale_or_mean(mean_string, input_info)

    for input in input_info:
        if input.is_image:
            input.scale = np.ones(3)
            input.mean = np.zeros(3)

            if input.name in scale_map:
                input.scale = scale_map[input.name]
            if input.name in mean_map:
                input.mean = mean_map[input.name]

    return input_info, reshape


def get_batch_size(inputs_info):
    batch_size = 0
    for info in inputs_info:
        batch_index = info.layout.index('N') if 'N' in info.layout else -1
        if batch_index != -1:
            if batch_size == 0:
                batch_size = len(info.shape[batch_index])
            elif batch_size != len(info.shape[batch_index]):
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
