# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import PosixPath, WindowsPath
from copy import deepcopy
import json

import numpy as np

from openvino.tools.pot.version import get_version
from .cpu_patterns import get_cpu_ignored_patterns, get_cpu_spr_ignored_patterns
from .gpu_patterns import get_gpu_ignored_patterns
from .npu_patterns import get_npu_ignored_patterns
from .gna_patterns import get_gna_ignored_patterns, get_gna3_ignored_patterns
from .special_operations import QUANTIZE_AGNOSTIC_OPERATIONS
from .node_utils import get_all_node_outputs, get_input_shape

HARDWARE_AWARE_IGNORED_PATTERNS = {
    'ANY': get_cpu_ignored_patterns(),
    'CPU': get_cpu_ignored_patterns(),
    'GPU': get_gpu_ignored_patterns(),
    'NPU': get_npu_ignored_patterns(),
    'GNA': get_gna_ignored_patterns(),
    'GNA3': get_gna3_ignored_patterns(),
    'GNA3.5': get_gna3_ignored_patterns(),
    'CPU_SPR': get_cpu_spr_ignored_patterns()
}

DEFAULT_PATH = 'PATH'

HARDWARE_SPECIAL_FIELDS = ['target_device', 'primary_bitwidth']


# pylint: disable=method-hidden
class PathEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (PosixPath, WindowsPath)):
            return DEFAULT_PATH

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)


def convert_mo_to_ie_operation(op):
    if 'IE' not in op:
        return None

    def process_list(op, value):
        ie_mo_naming = {}
        ie_mo_naming_loc = ie_mo_naming
        data_info = False
        for item in value:
            if data_info:
                # we write fields form 'data' to 'attributes'
                ie_mo_naming_loc = ie_mo_naming['attributes']
            if isinstance(item, list):
                ie_mo_naming_loc.update(process_list(op, item))
            elif isinstance(item, tuple):
                if len(item) == 2 and isinstance(item[0], str) and \
                        (callable(item[1]) or isinstance(item[1], str)):
                    if callable(item[1]) and item[1](op) is not None:
                        ie_mo_naming_loc[item[0]] = item[1](op)
                    elif item[1] in op and op[item[1]] is not None:
                        ie_mo_naming_loc[item[0]] = op[item[1]]
                else:
                    ie_mo_naming_loc.update(process_list(op, item))
            elif isinstance(item, str) and item in op and op[item] is not None:
                ie_mo_naming_loc[item] = op[item]
            elif item == 'data':
                data_info = True
                # we write fields form 'data' to 'attributes'
                ie_mo_naming['attributes'] = {}
        return ie_mo_naming

    return process_list(op, op['IE'])


def operations_matched(src_op, dst_op):
    def match_attrs(src_attrs, dst_attrs):
        for name, value in src_attrs.items():
            if name in dst_attrs:
                if isinstance(src_attrs[name], dict) and isinstance(dst_attrs[name], dict):
                    for k, v in src_attrs[name].items():
                        if k in dst_attrs[name] and dst_attrs[name][k] != v:
                            return False
                elif dst_attrs[name] != value:
                    return False
        return True

    ie_in_src_op = 'IE' in src_op
    ie_in_dst_op = 'IE' in dst_op

    if ie_in_src_op == ie_in_dst_op:
        return match_attrs(src_op, dst_op)

    ie_src_op = convert_mo_to_ie_operation(src_op) if ie_in_src_op else src_op
    ie_dst_op = convert_mo_to_ie_operation(dst_op) if ie_in_dst_op else dst_op
    return match_attrs(ie_src_op, ie_dst_op)


def find_operation_matches(src_ops, dst_ops):
    if not isinstance(src_ops, (list, tuple)):
        src_ops = [src_ops]
    if not isinstance(dst_ops, (list, tuple)):
        dst_ops = [dst_ops]

    result = []
    for src_op in src_ops:
        for dst_op in dst_ops:
            if operations_matched(src_op, dst_op):
                result.append((src_op, dst_op))
    return result


def get_operation_list(hardware_config):
    hw_ops = []
    for item in hardware_config:
        if any([special_value in item for special_value in HARDWARE_SPECIAL_FIELDS]):
            continue

        op = {}
        for key, value in item.items():
            if (key == 'attributes') or (not isinstance(value, dict)):
                op[key] = value
        if op not in hw_ops:
            hw_ops.append(op)
    return hw_ops

def get_operation_list_with_outputs(hardware_config):
    hw_ops = []
    for item in hardware_config:
        if any([special_value in item for special_value in HARDWARE_SPECIAL_FIELDS]):
            continue
        if 'quantization' in item and 'outputs' in item['quantization']:
            hw_ops.append(item['type'])
    return hw_ops

def create_quantization_info_for_mo(config):
    quantization_section = {}
    config_info = {key: config[key] for key in ['compression', 'engine']}
    quantization_section['config'] = json.dumps(config_info, indent='\t',
                                                cls=PathEncoder).replace('"', "'").replace('\n', '\n\t')
    quantization_section['version'] = get_version()
    return quantization_section


def create_cli_params_for_mo(args):
    cli_args = {key: value for key, value in args.__dict__.items() if key not in ['config']}
    cli_args['output_dir'] = DEFAULT_PATH
    return cli_args


def is_ignored(ignored_params, op, skipped=True):
    """
        Return whether node should be ignored by algo or not.
    """
    if ignored_params.get('skip_model') or \
            skipped and 'skipped' in op and op['skipped'] or\
            op.fullname in ignored_params['scope']:
        return True
    for operation in ignored_params['operations']:
        if op.type == operation['type']:
            if 'attributes' not in operation:
                return True
            for attribute in operation['attributes']:
                if attribute not in op or op[attribute] != operation['attributes'][attribute]:
                    return False
            return True
    return False


def get_hw_aware_ignored_patterns(target_device):
    return HARDWARE_AWARE_IGNORED_PATTERNS[target_device]


def preprocess_ignored_params(ignored_params, model):
    default_ignored_params = {
        'skip_model': False,
        'scope': [],
        'operations': []
    }
    if ignored_params is None:
        ignored_params = {}

    if not model.is_cascade:
        default_ignored_params.update(ignored_params)
        return check_agnostic_and_ignored_params(model, default_ignored_params)

    ignored_params_ = {}
    for model_dict in model.models:
        name = model_dict['name']
        ignored_params_[name] = deepcopy(default_ignored_params)
        ignored_params_[name].update(ignored_params.get(name, {}))
        ignored_params_[name]['scope'] = ['{}_{}'.format(name, node)
                                          for node in ignored_params_[name]['scope']]

    return check_agnostic_and_ignored_params(model, ignored_params_)


def check_agnostic_and_ignored_params(model, ignored_params):
    def add_new_ignored_params(model, node, quantize_agnostic, ignored_params, model_is_cascade):
        children = [node for node in get_all_node_outputs(node) if node is not None]
        for child in children:
            if child not in quantize_agnostic:
                ignored_params['scope'].append(child.fullname)
            else:
                add_new_ignored_params(model, node, quantize_agnostic,\
                                       ignored_params, model_is_cascade)
        return ignored_params

    quantize_agnostic = [op['type'] for op in QUANTIZE_AGNOSTIC_OPERATIONS]
    for model_dict in model.models:
        dict_ignored_operation_model = ignored_params[model_dict['name']] if model.is_cascade else ignored_params
        ignored_params_operation = [op['type'] for op in dict_ignored_operation_model['operations']]

        for node in model_dict['model'].get_op_nodes():
            if (node.type in ignored_params_operation or node.fullname in dict_ignored_operation_model['scope']) \
                                                                       and node.type in quantize_agnostic:

                new_ignored_params = add_new_ignored_params(model_dict['model'], node,
                                                            quantize_agnostic,
                                                            dict_ignored_operation_model,
                                                            model.is_cascade)

                if model.is_cascade:
                    ignored_params[model_dict['name']].update(new_ignored_params)
                else:
                    ignored_params = new_ignored_params

    return ignored_params


def is_data_type_quantizable(type_node):
    return type_node not in (np.int32, np.int64, bool)


def get_hardware_config_operation_type(node, available_types):
    """ This function gets type by child
    for hardware configuration of FQ node
    :param node: node-type object
    :param available_types: available types with config
    :return: default or special type of layer as string
    """

    def _is_depth_wise(node):
        if node.type == 'Convolution' and node.has_valid('group'):
            group = node['group']
            output = node['output']
            input_shape = get_input_shape(node, 0)
            if group == output and input_shape[1] == output:
                return True
        return False

    type_checkers = {
        'DepthWiseConvolution': _is_depth_wise
    }

    for real_type in type_checkers:
        if real_type in available_types and type_checkers[real_type](node):
            return real_type
    return node.type
