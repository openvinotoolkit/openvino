# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from datetime import timedelta
import enum
from openvino.runtime import PartialShape, Dimension, Layout, Type, serialize, properties, OVAny
from math import ceil

from .constants import DEVICE_DURATION_IN_SECS, UNKNOWN_DEVICE_TYPE, \
    AUTO_DEVICE_NAME, MULTI_DEVICE_NAME, HETERO_DEVICE_NAME
from .logging import logger

import json
import re
import numpy as np


import sys
import argparse

INPUT_OUTPUT_PRECISION_CHOICES = [
    'bool',
    'f16', 'f32', 'f64',
    'i8', 'i16', 'i32', 'i64',
    'u8', 'u16', 'u32', 'u64']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid positive int value")
    return ivalue


def check_positive_float(value):
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid positive int value")
    return ivalue


class print_help(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()
        sys.exit()


class HelpFormatterWithLines(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        lines = super()._split_lines(text, width)
        lines += ['']
        if "simple JSON file" not in text:
            return lines
        lines = text.split('\n')
        return lines


def parse_args():
    parser = argparse.ArgumentParser(
        add_help=False, formatter_class=HelpFormatterWithLines)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action=print_help, nargs='?', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--path_to_model', type=str, required=True,
                      help='Required. Path to an .xml file with a trained model')
    args.add_argument('-t', '--threshold', type=check_positive_float, required=False, default=0.01,
                      help='Optional. Acceptable threshold of accuracy loss')
    args.add_argument('-n', '--number_infer_requests', type=check_positive, required=False, default=100,
                      help='Optional. Number of infer requests run with random data. Default is 100')
    args.add_argument('-s', '--data_shape', type=str, required=False, default='',
                      help='Optional if model shapes are all static. '
                      'Required if at least one input shape is dynamic and input images are not provided.'
                      'Set shape for input tensors. For example, "input1[1,3,224,224][1,3,448,448],input2[1,4][1,8]" or "[1,3,224,224][1,3,448,448] in case of one input size.')
    # args.add_argument('-p', '--infer_precision', type=str, required=False,
    #                   help='Optional. Specifies the inference precision. Example #1: \'-infer_precision bf16\'. Example #2: \'-infer_precision CPU:bf16,GPU:f32\'')

    parsed_args = parser.parse_args()

    return parsed_args, parser

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def percentile(values, percent):
    return values[ceil(len(values) * percent / 100) - 1]


can_run_bf16_ops = {"Convolution", "FullyConnected", "RNNCell", "RNNSeq",
                    "MatMul", "MulAdd", "Add", "ROIPooling", "Interpolate",
                    "MVN", "MaxPool"}


def generate_bf16_ops_list(bf16list):
    ops_list = []
    for it in bf16list:
        # print(f"item={it}")
        if it[1] in can_run_bf16_ops:
            ops_list.append(it[0])
    return ops_list

def get_element_type(precision):
    format_map = {
      'bool'    : Type.boolean,

      'f16'  : Type.f16,
      'f32'  : Type.f32,
      'f64'  : Type.f64,

      'i8'   : Type.i8,
      'i16'  : Type.i16,
      'i32'  : Type.i32,
      'i64'  : Type.i64,

      'u8'   : Type.u8,
      'u16'  : Type.u16,
      'u32'  : Type.u32,
      'u64'  : Type.u64,
    }
    if precision in format_map.keys():
        return format_map[precision]

    raise Exception(f"Undefined precision: '{precision}' !")


# # # def fuse_mean_scale(preproc: PrePostProcessor, app_inputs_info):
# # #     # TODO: remove warning after 23.3 release
# # #     warned = False
# # #     warn_msg = 'Mean/scale values are fused into the model. This slows down performance compared to --imean and --iscale which existed before'
# # #     for input_info in app_inputs_info:
# # #         if input_info.mean.size:
# # #             if not warned:
# # #                 logger.warning(warn_msg)
# # #                 warned = True
# # #             preproc.input(input_info.name).preprocess().convert_element_type(Type.f32).mean(input_info.mean)
# # #         if input_info.scale.size:
# # #             if not warned:
# # #                 logger.warning(warn_msg)
# # #                 warned = True
# # #             preproc.input(input_info.name).preprocess().convert_element_type(Type.f32).scale(input_info.scale)


# # # def pre_post_processing(model: Model, app_inputs_info, input_precision: str, output_precision: str, input_output_precision: str):
# # #     pre_post_processor = PrePostProcessor(model)
# # #     if input_precision:
# # #         element_type = get_element_type(input_precision)
# # #         for i in range(len(model.inputs)):
# # #             pre_post_processor.input(i).tensor().set_element_type(element_type)
# # #             app_inputs_info[i].element_type = element_type
# # #     if output_precision:
# # #         element_type = get_element_type(output_precision)
# # #         for i in range(len(model.outputs)):
# # #             pre_post_processor.output(i).tensor().set_element_type(element_type)
# # #     user_precision_map = {}
# # #     if input_output_precision:
# # #         user_precision_map = parse_input_output_precision(input_output_precision)
# # #         input_names = get_input_output_names(model.inputs)
# # #         input_node_names = get_node_names(model.inputs)
# # #         output_names = get_input_output_names(model.outputs)
# # #         output_node_names = get_node_names(model.outputs)
# # #         for node_name, precision in user_precision_map.items():
# # #             user_precision_map[node_name] = get_element_type(precision)
# # #         for name, element_type in user_precision_map.items():
# # #             if name in input_names or name in input_node_names:
# # #                 input_index = input_names.index(name) if name in input_names else input_node_names.index(name)
# # #                 app_inputs_info[input_index].element_type = element_type
# # #                 pre_post_processor.input(input_index).tensor().set_element_type(element_type)
# # #             elif name in output_names or name in output_node_names:
# # #                 if name in output_names:
# # #                     pre_post_processor.output(name).tensor().set_element_type(element_type)
# # #                 else:
# # #                     pre_post_processor.output(output_node_names.index(name)).tensor().set_element_type(element_type)
# # #             else:
# # #                 raise Exception(f"Node '{name}' does not exist in model")

# # #     # update app_inputs_info
# # #     if not input_precision:
# # #         inputs = model.inputs
# # #         input_node_names = get_node_names(model.inputs)
# # #         for i in range(len(inputs)):
# # #             if app_inputs_info[i].name in user_precision_map:
# # #                 app_inputs_info[i].element_type = user_precision_map[app_inputs_info[i].name]
# # #             elif input_node_names[i] in user_precision_map:
# # #                 app_inputs_info[i].element_type = user_precision_map[input_node_names[i]]
# # #             elif app_inputs_info[i].is_image:
# # #                 app_inputs_info[i].element_type = Type.u8
# # #                 pre_post_processor.input(i).tensor().set_element_type(Type.u8)

# # #     fuse_mean_scale(pre_post_processor, app_inputs_info)

# # #     # set layout for model input
# # #     for info in app_inputs_info:
# # #         pre_post_processor.input(info.name).model().set_layout(info.layout)

# # #     model = pre_post_processor.build()


# # # def parse_input_output_precision(arg_map: str):
# # #     arg_map = arg_map.replace(" ", "")
# # #     pairs = [x.strip() for x in arg_map.split(',')]

# # #     parsed_map = {}
# # #     for pair in pairs:
# # #         key_value = [x.strip() for x in pair.split(':')]
# # #         name, precision = key_value[0], key_value[1]
# # #         # input's name can contain ':'
# # #         if len(key_value) == 3:
# # #             name = key_value[0] + ':' + key_value[1]
# # #             precision = key_value[2]
# # #         parsed_map.update({name:precision})

# # #     return parsed_map


# # # def print_inputs_and_outputs_info(model: Model):
# # #     inputs = model.inputs
# # #     logger.info("Model inputs:")
# # #     for input in inputs:
# # #         in_name = " , ".join(input.get_names())
# # #         node_name = input.node.get_friendly_name()

# # #         if in_name=="": in_name = "***NO_NAME***"
# # #         if node_name=="": node_name = "***NO_NAME***"

# # #         logger.info(f"    {in_name} (node: {node_name}) : {input.element_type.get_type_name()} / "
# # #                     f"{str(input.node.layout)} / {input.partial_shape}")

# # #     outputs = model.outputs
# # #     logger.info("Model outputs:")
# # #     for output in outputs:
# # #         out_name = " , ".join(output.get_names())
# # #         node_name = output.get_node().input(0).get_source_output().get_node().get_friendly_name()

# # #         if out_name=="": out_name = "***NO_NAME***"
# # #         if node_name=="": node_name = "***NO_NAME***"

# # #         logger.info(f"    {out_name} (node: {node_name}) : {output.element_type.get_type_name()} / "
# # #                     f"{str(output.node.layout)} / {output.partial_shape}")


# # # def get_number_iterations(number_iterations: int, nireq: int, num_shapes: int, api_type: str):
# # #     niter = number_iterations

# # #     if api_type == 'async' and niter:
# # #         if num_shapes > nireq:
# # #             niter = int(((niter + num_shapes -1) / num_shapes) * num_shapes)
# # #             if number_iterations != niter:
# # #                 logger.warning('Number of iterations was aligned by number of input shapes '
# # #                             f'from {number_iterations} to {niter} using number of possible input shapes {num_shapes}')
# # #         else:
# # #             niter = int((niter + nireq - 1) / nireq) * nireq
# # #             if number_iterations != niter:
# # #                 logger.warning('Number of iterations was aligned by request number '
# # #                             f'from {number_iterations} to {niter} using number of requests {nireq}')

# # #     return niter

# # # def get_duration_seconds(time, number_iterations, device):
# # #     if time:
# # #         # time limit
# # #         return time

# # #     if not number_iterations:
# # #         return get_duration_in_secs(device)
# # #     return 0


# # class LatencyGroup:
# #     def __init__(self, input_names, input_shapes):
# #         self.input_names = input_names
# #         self.input_shapes = input_shapes
# #         self.times = list()
# #         self.median = 0.
# #         self.avg = 0.
# #         self.min = 0.
# #         self.max = 0.

# #     def __str__(self):
# #         return str().join(f" {name}: {str(shape)}" for name, shape in zip(self.input_names, self.input_shapes))


# # def get_latency_groups(app_input_info):
# #     num_groups = max(len(info.shapes) for info in app_input_info)
# #     latency_groups = []
# #     for i in range(num_groups):
# #         names = list()
# #         shapes = list()
# #         for info in app_input_info:
# #             names.append(info.name)
# #             shapes.append(info.shapes[i % len(info.shapes)])
# #         latency_groups.append(LatencyGroup(names, shapes))
# #     return latency_groups


# # def get_duration_in_milliseconds(duration):
# #     return duration * 1000


# # def get_duration_in_secs(target_device):
# #     duration = 0
# #     for device in DEVICE_DURATION_IN_SECS:
# #         if device in target_device:
# #             duration = max(duration, DEVICE_DURATION_IN_SECS[device])

# #     if duration == 0:
# #         duration = DEVICE_DURATION_IN_SECS[UNKNOWN_DEVICE_TYPE]
# #         logger.warning(f'Default duration {duration} seconds is used for unknown device {target_device}')

# #     return duration


# def check_for_static(app_input_info):
#     for info in app_input_info:
#         if info.is_dynamic:
#             return False
#     return True


# def can_measure_as_static(app_input_info):
#     for info in app_input_info:
#         if len(info.shapes) > 1:
#             return False
#     return True

# # meta_plugins = [ MULTI_DEVICE_NAME, HETERO_DEVICE_NAME, AUTO_DEVICE_NAME ]
# # def is_virtual_device(device_name) -> bool:
# #     return device_name in meta_plugins

# # def is_virtual_device_found(device_names) -> bool:
# #     return any(is_virtual_device(device_name) for device_name in device_names)

# def parse_devices(device_string):
#     result = []
#     target_device = device_string.partition(":")[0]
#     result.append(target_device)
#     if device_string.find(":") != -1:
#         hw_devices_str = device_string.partition(":")[-1]
#         for hw_device in hw_devices_str.split(','):
#             if hw_device[0] == '-':
#                 hw_device = hw_device[1:]
#             result.append(hw_device)
#     return result

# def parse_value_per_device(devices, values_string, value_type):
#     # Format: <device1>:<value1>,<device2>:<value2> or just <value>
#     result = {}
#     if not values_string:
#         return result
#     device_value_strings = values_string.split(',')
#     for device_value_string in device_value_strings:
#         device_value_vec = device_value_string.split(':')
#         if len(device_value_vec) == 2:
#             device_name = device_value_vec[0]
#             value = device_value_vec[1]
#             if device_name in devices:
#                 result[device_name] = value
#             else:
#                 devices_str = ""
#                 for device in devices:
#                     devices_str += device + " "
#                 devices_str = devices_str.strip()
#                 raise Exception(f"Failed to set property to '{device_name}' " \
#                                 f"which is not found in the target devices list '{devices_str}'!")
#         elif len(device_value_vec) == 1:
#             value = device_value_vec[0]
#             for device in devices:
#                 result[device] = value
#         elif not device_value_vec:
#             raise Exception('Unknown string format: ' + values_string)
#     return result

# def parse_value_for_virtual_device(device, values_string):
#     isExist = device in values_string.keys()
#     if isExist and len(values_string) > 1:
#         if device == MULTI_DEVICE_NAME:
#             # Remove the element that the key is virtual device MULTI
#             # e.g. MULTI:xxx -nstreams 2 will set nstreams 2 to xxx.
#             values_string.pop(device)
#         elif device == AUTO_DEVICE_NAME or device == HETERO_DEVICE_NAME:
#             # Just keep the element that the key is virtual device AUTO/HETERO
#             # e.g. AUTO:xxx,xxx -nstreams 2 will trigger exception that AUTO plugin didn't support nstream property.
#             value = values_string.get(device)
#             values_string.clear()
#             values_string[device] = value
#     keys = values_string.keys()
#     for key in list(values_string):
#         if device not in list(values_string):
#             values_string[device] = '{'
#         else:
#             values_string[device] += ','
#         values_string[device] += key + ":" + values_string.get(key)
#         del values_string[key]
#     if device in values_string.keys():
#         values_string[device] = values_string[device].strip()
#         if values_string[device] != '':
#             values_string[device] += '}'
#     return

def process_help_inference_string(runner, device_number_streams):
    output_string = f'Start inference asynchronously'
    return output_string

def dump_exec_graph(compiled_model, model_path):
    serialize(compiled_model.get_runtime_model(), model_path)

def print_perf_counters_sort(perf_counts_list,sort_flag="sort"):
    """ Print opts time cost and can be sorted according by each opts time cost
    """
    for ni in range(len(perf_counts_list)):
        perf_counts = perf_counts_list[ni]
        total_time = timedelta()
        total_time_cpu = timedelta()
        logger.info(f"Performance counts sorted for {ni}-th infer request")
        for pi in perf_counts:
            total_time += pi.real_time
            total_time_cpu += pi.cpu_time

        total_time = total_time.microseconds
        total_time_cpu = total_time_cpu.microseconds
        total_real_time_proportion = 0
        total_detail_data=[]
        for pi in perf_counts:
            node_name = pi.node_name
            layerStatus = pi.status
            layerType = pi.node_type
            real_time = pi.real_time.microseconds
            cpu_time = pi.cpu_time.microseconds
            real_proportion = round(real_time/total_time,4)
            execType = pi.exec_type
            tmp_data=[node_name,layerStatus,layerType,real_time,cpu_time,real_proportion,execType]
            total_detail_data.append(tmp_data)
            total_real_time_proportion += real_proportion
        total_detail_data = np.array(total_detail_data)
        if sort_flag=="sort":
            total_detail_data = sorted(total_detail_data,key=lambda tmp_data:tmp_data[-4],reverse=True)
        elif sort_flag=="no_sort":
            total_detail_data = total_detail_data
        elif sort_flag=="simple_sort":
            total_detail_data = sorted(total_detail_data,key=lambda tmp_data:tmp_data[-4],reverse=True)
            total_detail_data = [tmp_data for tmp_data in total_detail_data if str(tmp_data[1])!="Status.NOT_RUN"]
        print_detail_result(total_detail_data)
        print(f'Total time:       {total_time / 1000:.3f} milliseconds')
        print(f'Total CPU time:   {total_time_cpu / 1000:.3f} milliseconds')
        print(f'Total proportion: {"%.2f"%(round(total_real_time_proportion)*100)} % \n')
    return total_detail_data

def print_detail_result(result_list):
    """ Print_perf_counters_sort result
    """
    max_print_length = 20
    for tmp_result in result_list:
        node_name = tmp_result[0]
        layerStatus = tmp_result[1]
        layerType = tmp_result[2]
        real_time = tmp_result[3]
        cpu_time = tmp_result[4]
        real_proportion = "%.2f" % (tmp_result[5] * 100)
        if real_proportion == "0.00":
            real_proportion = "N/A"
        execType = tmp_result[6]
        print(f"{node_name[:max_print_length - 4] + '...' if (len(node_name) >= max_print_length) else node_name:<20} "
                f"{str(layerStatus):<20} "
                f"layerType: {layerType[:max_print_length - 4] + '...' if (len(layerType) >= max_print_length) else layerType:<20} "
                f"execType: {execType:<20} "
                f"realTime (ms): {real_time / 1000:<10.3f} "
                f"cpuTime (ms): {cpu_time / 1000:<10.3f}"
                f"proportion: {str(real_proportion +'%'):<8}")

def print_perf_counters(perf_counts_list):
    max_print_length = 20
    for ni in range(len(perf_counts_list)):
        perf_counts = perf_counts_list[ni]
        total_time = timedelta()
        total_time_cpu = timedelta()
        logger.info(f"Performance counts for {ni}-th infer request")
        for pi in perf_counts:
            print(f"{pi.node_name[:max_print_length - 4] + '...' if (len(pi.node_name) >= max_print_length) else pi.node_name:<20} "
                f"{str(pi.status):<20} "
                f"layerType: {pi.node_type[:max_print_length - 4] + '...' if (len(pi.node_type) >= max_print_length) else pi.node_type:<20} "
                f"execType: {pi.exec_type:<20} "
                f"realTime (ms): {pi.real_time / timedelta(milliseconds=1):<10.3f} "
                f"cpuTime (ms): {pi.cpu_time / timedelta(milliseconds=1):<10.3f}")

            total_time += pi.real_time
            total_time_cpu += pi.cpu_time
        print(f'Total time:     {total_time / timedelta(milliseconds=1)} milliseconds')
        print(f'Total CPU time: {total_time_cpu / timedelta(milliseconds=1)} milliseconds\n')

def parser_perf_counters_sort(perf_counts_list):
    perf_counts = perf_counts_list[0]
    total_detail_data = []
    for pi in perf_counts:
        node_name = pi.node_name
        layerType = pi.node_type
        real_time = pi.real_time.microseconds
        execType = pi.exec_type
        tmp_data=[node_name,layerType,real_time,execType]
        total_detail_data.append(tmp_data)
    total_detail_data = np.array(total_detail_data)
    total_detail_data = sorted(total_detail_data,key=lambda tmp_data:tmp_data[-2],reverse=False)
    return total_detail_data

def parser_perf_counters(perf_counts_list):
    layerlist = []
    total_time = timedelta()
    total_time_cpu = timedelta()
    perf_counts = perf_counts_list[0]
    for pi in perf_counts:
        total_time += pi.real_time
        total_time_cpu += pi.cpu_time
        if pi.exec_type.find("bf16") >=0 :
            layerlist.append([pi.node_name, pi.node_type, pi.exec_type])
            # print(f"### {pi.node_name} {pi.node_type} {pi.exec_type}")
    average_time = total_time / timedelta(milliseconds=1) / len(perf_counts)
    return layerlist, average_time

# def get_command_line_arguments(argv):
#     parameters = []
#     arg_name = ''
#     arg_value = ''
#     for arg in argv[1:]:
#         if '=' in arg:
#             arg_name, arg_value = arg.split('=')
#             parameters.append((arg_name, arg_value))
#             arg_name = ''
#             arg_value = ''
#         else:
#           if arg[0] == '-':
#               if arg_name != '':
#                 parameters.append((arg_name, arg_value))
#                 arg_value = ''
#               arg_name = arg
#           else:
#               arg_value = arg
#     if arg_name != '':
#         parameters.append((arg_name, arg_value))
#     return parameters


def get_input_output_names(ports):
    return [port.any_name for port in ports]

def get_node_names(ports):
    return [port.node.friendly_name for port in ports]

def get_data_shapes_map(data_shape_string, input_names):
    # Parse parameter string like "input0[shape1][shape2],input1[shape1]" or "[shape1][shape2]" (applied to all inputs)
    return_value = {}
    if data_shape_string:
        data_shape_string += ','
        matches = re.findall(r'(.*?\[.*?\]),', data_shape_string)
        if matches:
            for match in matches:
                input_name = match[:match.find('[')]
                shapes = re.findall(r'\[(.*?)\]', match[len(input_name):])
                if input_name:
                    return_value[input_name] = list(PartialShape(shape_str) for shape_str in shapes)
                else:
                    data_shapes = list(PartialShape(shape_str) for shape_str in shapes)
                    num_inputs, num_shapes = len(input_names), len(data_shapes)
                    if num_shapes != 1 and num_shapes % num_inputs != 0:
                        raise Exception(f"Number of provided data_shapes is not a multiple of the number of model inputs!")
                    return_value = defaultdict(list)
                    for i in range(max(num_shapes, num_inputs)):
                        return_value[input_names[i % num_inputs]].append(data_shapes[i % num_shapes])
                    return return_value
        else:
            raise Exception(f"Can't parse input parameter: {data_shape_string}")
    return return_value

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


# def parse_scale_or_mean(parameter_string, input_info):
#     # Parse parameter string like "input0[value0],input1[value1]" or "[value]" (applied to all inputs)
#     return_value = {}
#     if parameter_string:
#         matches = re.findall(r'(.*?)\[(.*?)\],?', parameter_string)
#         if matches:
#             for match in matches:
#                 input_name, value = match
#                 f_value = np.array(value.split(",")).astype(float)
#                 if input_name != '':
#                     return_value[input_name] = f_value
#                 else:
#                     for input in input_info:
#                         if input.is_image:
#                             return_value[input.name] = f_value
#         else:
#             raise Exception(f"Can't parse input parameter: {parameter_string}")
#     return return_value

class AppInputInfo:
    def __init__(self):
        self.element_type = None
        self.layout = Layout()
        self.partial_shape = None
        self.data_shapes = []
        self.scale = np.empty([0])
        self.mean = np.empty([0])
        self.name = None
        self.node_name = None

    @property
    def is_image(self):
        if str(self.layout) not in [ "[N,C,H,W]", "[N,H,W,C]", "[C,H,W]", "[H,W,C]" ]:
            return False
        return self.channels == 3

    @property
    def is_image_info(self):
        if str(self.layout) != "[N,C]":
            return False
        return len(self.channels) >= 2 if self.channels.is_static else self.channels.relaxes(Dimension(2))

    def getDimensionByLayout(self, character):
        if self.layout.has_name(character):
            return self.partial_shape[self.layout.get_index_by_name(character)]
        else:
            return Dimension(0)

    def getDimensionsByLayout(self, character):
        if self.layout.has_name(character):
            d_index = self.layout.get_index_by_name(character)
            dims = []
            for shape in self.data_shapes:
                dims.append(shape[d_index])
            return dims
        else:
            return [0] * len(self.data_shapes)

    @property
    def shapes(self):
        if self.is_static:
            return [self.partial_shape.to_shape()]
        else:
            return self.data_shapes

    @property
    def width(self):
        return len(self.getDimensionByLayout("W"))

    @property
    def widths(self):
        return self.getDimensionsByLayout("W")

    @property
    def height(self):
        return len(self.getDimensionByLayout("H"))

    @property
    def heights(self):
        return self.getDimensionsByLayout("H")

    @property
    def channels(self):
        return self.getDimensionByLayout("C")

    @property
    def is_static(self):
        return self.partial_shape.is_static

    @property
    def is_dynamic(self):
        return self.partial_shape.is_dynamic


def get_inputs_info(data_shape_string, batch_size, inputs):
    input_names = get_input_output_names(inputs)
    input_node_names = get_node_names(inputs)
    data_shape_map = get_data_shapes_map(data_shape_string, input_names)
    batch_size = Dimension(batch_size)
    input_info = []
    for i in range(len(inputs)):
        info = AppInputInfo()
        # Input name
        info.name = input_names[i]
        # Input node name
        info.node_name = input_node_names[i]
        # Input precision
        info.element_type = inputs[i].element_type
        # Shape
        info.partial_shape = inputs[i].partial_shape

        # Layout
        image_colors_dim_max = 4
        shape = info.partial_shape
        num_dims = len(shape)
        if num_dims == 4:
            if shape[1].get_max_length() <= image_colors_dim_max and shape[3].get_max_length() > image_colors_dim_max:
                info.layout = Layout("NCHW")
            elif shape[3].get_max_length() <= image_colors_dim_max and shape[1].get_max_length() > image_colors_dim_max:
                info.layout = Layout("NHWC")
        elif num_dims == 3:
            if shape[0].get_max_length() <= image_colors_dim_max and shape[2].get_max_length() > image_colors_dim_max:
                info.layout = Layout("CHW")
            elif shape[2].get_max_length() <= image_colors_dim_max and shape[0].get_max_length() > image_colors_dim_max:
                info.layout = Layout("HWC")

        # Data shape
        if (info.name in data_shape_map or info.node_name in data_shape_map) and info.is_dynamic:
            used_name = info.name if info.name in data_shape_map else info.node_name
            for p_shape in data_shape_map[used_name]:
                if p_shape.is_dynamic:
                    raise Exception(f"Data shape always should be static, {str(p_shape)} is dynamic.")
                elif info.partial_shape.compatible(p_shape):
                    info.data_shapes.append(p_shape.to_shape())
                else:
                    raise Exception(f"Data shape '{str(p_shape)}' provided for input '{info.name}' "
                                    f"is not compatible with partial shape '{str(info.partial_shape)}' for this input.")
        elif info.name in data_shape_map or input_node_names[i] in data_shape_map:
            logger.warning(f"Input '{info.name}' has static shape. Provided data shapes for this input will be ignored.")

        input_info.append(info)

    return input_info


# def get_network_batch_size(inputs_info):
#     null_dimension = Dimension(0)
#     batch_size = null_dimension
#     for info in inputs_info:
#         batch_index = info.layout.get_index_by_name('N') if info.layout.has_name('N') else -1
#         if batch_index != -1:
#             if batch_size == null_dimension:
#                 batch_size = info.partial_shape[batch_index]
#             elif batch_size != info.partial_shape[batch_index]:
#                 raise Exception("Can't deterimine batch size: batch is different for different inputs!")
#     if batch_size == null_dimension:
#         batch_size = Dimension(1)
#     return batch_size


# def show_available_devices():
#     print("\nAvailable target devices:  ", ("  ".join(Core().available_devices)))


def device_properties_to_string(config):
    ret = "{"
    for k, v in config.items():
        if isinstance(v, dict):
            sub_str = "{"
            for sk, sv in v.items():
                if isinstance(sv, bool):
                    sv = "YES" if sv else "NO"
                if isinstance(sv, properties.Affinity):
                    sv = sv.name
                sub_str += "{0}:{1},".format(sk, sv)
            sub_str = sub_str[:-1]
            sub_str += "}"
            ret += "{0}:{1},".format(k, sub_str)
        else:
            ret += "{0}:{1},".format(k, v)
    ret = ret[:-1]
    ret += "}"
    return ret


def string_to_device_properties(device_properties_str):
    ret = {}
    if not device_properties_str:
        return ret
    if not device_properties_str.startswith("{") or not device_properties_str.endswith("}"):
        raise Exception(
            "Failed to parse device properties. Value of device properties should be started with '{' and ended with '}'."
            "They are actually {} and {}".format(device_properties_str[0], device_properties_str[-1]))
    pattern = r'(\w+):({.+?}|[^,}]+)'
    pairs = re.findall(pattern, device_properties_str)
    for key, value in pairs:
        if value.startswith("{") and value.endswith("}"):
            value = value[1:-1]
            nested_pairs = re.findall(pattern, value)
            nested_dict = {}
            for nested_key, nested_value in nested_pairs:
                nested_dict[nested_key] = nested_value
            value = nested_dict
        ret[key] = value
    return ret


def dump_config(filename, config):
    json_config = {}
    for device_name, device_config in config.items():
        json_config[device_name] = {}
        for key, value in device_config.items():
            if isinstance(value, OVAny) and (isinstance(value.value, dict)):
                value_string = device_properties_to_string(value.get())
            elif isinstance(value, (properties.hint.PerformanceMode, properties.Affinity)):
                value_string = value.name
            elif isinstance(value, OVAny):
                value_string = str(value.value)
            else:
                value_string = str(value)
                if isinstance(value, bool):
                    value_string = "YES" if value else "NO"
            json_config[device_name][key] = value_string

    with open(filename, 'w') as f:
        json.dump(json_config, f, indent=4)


def load_config(filename, config):
    with open(filename) as f:
        original_config = json.load(f)
    for device in original_config:
        config[device] = {}
        for property_name in original_config[device]:
            property_value = original_config[device][property_name]
            if property_name == properties.device.properties():
                property_value = string_to_device_properties(property_value)
            elif property_value in ("YES", "NO"):
                property_value = True if property_value == "YES" else False
            config[device][property_name] = OVAny(property_value)
