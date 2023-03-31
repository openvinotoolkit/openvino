# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import deque, defaultdict
from copy import deepcopy

from .range_estimator import get_range_estimator_config
from .utils import load_hardware_config
from ...graph.special_operations import QUANTIZE_AGNOSTIC_OPERATIONS, CONCAT_UNIFY_OUTPUTS,\
    CONCAT_UNIFY_INPUTS, TYPES_TO_QUANTIZABLE_PORTS
from ...graph.utils import find_operation_matches, get_operation_list, is_data_type_quantizable,\
    get_hardware_config_operation_type
from ...graph.model_utils import get_nodes_by_type, get_node_by_name
from ...graph.node_utils import get_input_shape, get_all_node_outputs,\
get_node_input, get_node_inputs, get_node_data_type, check_const_input, reset_node_fullname
from ...graph.passes import traverse_graph
from ...utils.logger import get_logger

logger = get_logger(__name__)


QUANTIZATION_PARAMETERS = [
    'level_high',
    'level_low',
    'mode',
    'granularity',
    'bits'
]

ACTIVATION_QUANTIZATION_MODES = ['activations', 'outputs']
QUANTIZATION_MODES = ['weights'] + ACTIVATION_QUANTIZATION_MODES


def get_fake_quantize_configuration(config):
    """ Create fake quantization configuration from the tool configuration
    :param config: dictionary with compression section from toolkit config file
    :return dictionary with fake quantization configuration
     """
    q_config = {mode: {} for mode in QUANTIZATION_MODES}
    for op_type, q_params in q_config.items():
        op_type_config = config.get(op_type, {})
        for param_name, param_value in op_type_config.items():
            if param_name in QUANTIZATION_PARAMETERS:
                q_params[param_name] = param_value
    return q_config


def intersect_configs(left, right, primary_bitwidth=None):
    """ intersect two sets of configurations """
    def _get_main_param_for_config(config):
        """ check main parameters intersection """
        main_params = {}
        for field_name in ['mode', 'granularity']:
            main_params[field_name] = config[field_name]
        return main_params

    def _intersect_configs(left_, right_):
        """ intersect two sets of configurations """
        result = []
        offset = 0
        for l_ in left_:
            l_main = _get_main_param_for_config(l_)
            for idx, r_ in enumerate(right_[offset:]):
                r_main = _get_main_param_for_config(r_)
                if l_main == r_main:
                    if primary_bitwidth and l_['bits'] == primary_bitwidth:
                        result.append(l_)
                    elif primary_bitwidth and r_['bits'] == primary_bitwidth:
                        result.append(r_)
                    elif l_['bits'] <= r_['bits']:
                        result.append(l_)
                    else:
                        result.append(r_)
                    offset += idx
                    break
        return result

    def _extend_configs(config):
        """ extend the set of configurations by adding configurations expressed
        through configurations from the input configuration set """
        def _insert_to_front(config_, item_):
            if item_ not in config_:
                config_.insert(0, item_)

        config_ext = []
        for item in reversed(config):
            _insert_to_front(config_ext, item)
            mode_asymmetric = 'mode' in item and item['mode'] == 'asymmetric'
            granularity_perchannel = 'granularity' in item and item['granularity'] == 'perchannel'
            if granularity_perchannel:
                item_ext = deepcopy(item)
                item_ext['granularity'] = 'pertensor'
                _insert_to_front(config_ext, item_ext)
            if mode_asymmetric:
                item_ext = deepcopy(item)
                item_ext['mode'] = 'symmetric'
                _insert_to_front(config_ext, item_ext)
            if mode_asymmetric and granularity_perchannel:
                item_ext = deepcopy(item)
                item_ext['mode'] = 'symmetric'
                item_ext['granularity'] = 'pertensor'
                _insert_to_front(config_ext, item_ext)
        return config_ext

    res = _intersect_configs(left, right)
    if not res:
        left_ext = _extend_configs(left)
        right_ext = _extend_configs(right)
        res = _intersect_configs(left_ext, right_ext)
    return res


def read_all_fake_quantize_configurations(config, hardware_config, model):
    """ Read all fake quantize configurations from hardware config which are suitable to
    every fake quantize node based on toolkit config file and sub graph of every fake quantize node
    :param config: dictionary with compression section from toolkit config file
    :param hardware_config: dictionary with hardware config
    :param model: CompressedModel instance to quantize
    :return dictionary with fake quantize names as keys and
     list of corresponding configurations as values
     """
    def _is_subset(left: dict, right: dict):
        """ Checks that x is a subset of y
        :param left: supposed to be subset of set 'right'
        :param right: set to check that left belongs to"""
        for key in left.keys():
            if key not in right.keys() or\
                    left[key] != right[key]:
                return False
        return True

    def _find_configurations(fq_name_, fq_type_):
        res_conf = []
        for op in ops:
            if fq_type_ in op['quantization']:
                confs = [conf for conf in op['quantization'][fq_type_]
                         if _is_subset(q_config[fq_type_], conf)]
                if confs:
                    res_conf = intersect_configs(res_conf, confs, primary_bitwidth) if res_conf else confs
                else:
                    logger.warning('Fake quantize node %s does not support configuration '
                                   'from tool config file (mismatch with hardware config)',
                                   fq_name_)
                    res_conf = intersect_configs(res_conf, q_config[fq_type_], primary_bitwidth) \
                        if res_conf else [q_config[fq_type_]]
                if not res_conf:
                    raise Exception('Fake quantize configuration cannot be empty')
        return res_conf

    q_config = get_fake_quantize_configuration(config)
    primary_bitwidth = hardware_config[1]['primary_bitwidth']

    res_fq_to_hw_conf = {}
    for fq_name, (types, fq_group) in _fake_quantize_to_types(model, hardware_config).items():
        res_fq_to_hw_conf[fq_name] = {fq_group: []}
        for type_ in types:
            child_name, op_type = type_
            ops = [op for op in hardware_config if op_type == op['type']]
            conf = _find_configurations(fq_name, fq_group)
            if conf:
                res_fq_to_hw_conf[fq_name][fq_group].append((child_name, conf))
    return res_fq_to_hw_conf


def add_range_estimator_configs(fq_to_hw_confs, config):
    """ Expand fake quantize configuration with range_estimator config
    :param fq_to_hw_confs: dictionary with fake quantize names as keys and its configurations as values
    :param config: tool config used to create range_estimator config
    :return dictionary with fake quantize nodes names as keys and its configurations as values
     extended with range_estimator config"""
    for confs in fq_to_hw_confs.values():
        for i_type, conf in confs.items():
            conf['range_estimator'] = get_range_estimator_config(config, i_type, conf['granularity'], conf['mode'])
    return fq_to_hw_confs


def get_configurations_by_preset(config, model, fq_to_hw_confs, hardware_config):
    """ Choose fake quantize configuration by preset
    :param config: dictionary with params algo section from toolkit config
    :param model: CompressedModel instance
    :param fq_to_hw_confs: dictionary with fake quantize names as keys and
     list of its configurations as values (read_all_fake_quantize_configurations(..) return value)
    :return dictionary with fake quantize nodes names as keys and
     suitable configuration chose by preset as values"""

    def _apply_preset_rule(preset_, fq_name, param_type, confs, to_skip=None):
        if param_type == 'weights':
            if preset_ == 'accuracy':
                return confs[-1]
            return confs[0]
        if not to_skip or fq_name not in [fq for _, fqs in to_skip for fq in fqs]:
            if preset_ == 'performance':
                return confs[0]
            return confs[-1]
        return confs

    def _intersect_and_apply_preset(preset_, fq_to_hw_confs_, fqs_to_unify_):

        def _unify_and_apply_preset(preset_, cur_conf, fqs_to_unify_):
            def _test_shapes(shapes):
                return any([s[0] != shapes[0][0] or len(s) == 1 or s[1] != shapes[0][1] for s in shapes])

            for bridges, fqs in fqs_to_unify_:
                res_conf = []
                with_concat = 'Concat' in [get_node_by_name(model, bridge).type for bridge in bridges]
                fq_input_shapes = [get_input_shape(get_node_by_name(model, fq), 0) for fq in fqs]
                unclear_layout = _test_shapes(fq_input_shapes)
                bridge_layers = [get_node_by_name(model, bridge) for bridge in bridges]
                bridge_input_shapes = [get_input_shape(layer, i) for layer in bridge_layers for i in layer.in_ports()]
                broadcasting = _test_shapes(bridge_input_shapes)
                for fq in fqs:
                    for key in cur_conf[fq]:
                        if key in ACTIVATION_QUANTIZATION_MODES:
                            if with_concat or unclear_layout or broadcasting:
                                if not isinstance(cur_conf[fq]['activations'], list):
                                     cur_conf[fq]['activations'] = [cur_conf[fq]['activations']]

                                configuration = [c for c in cur_conf[fq][key] if c['granularity'] == 'pertensor']
                            else:
                                configuration = cur_conf[fq][key]
                    res_conf = intersect_configs(res_conf, configuration, primary_bitwidth) if res_conf \
                        else configuration
                if not res_conf:
                    raise Exception('Fake quantize nodes {} cannot be unified'.format(fqs))
                for fq in fqs:
                    for key in cur_conf[fq]:
                        if key in ACTIVATION_QUANTIZATION_MODES:
                            cur_conf[fq][key] = _apply_preset_rule(preset_, fq, key, res_conf)
            return cur_conf

        primary_bitwidth = hardware_config[1]['primary_bitwidth']
        res = {}
        for key, value in fq_to_hw_confs_.items():
            conf = dict()
            for i_type in QUANTIZATION_MODES:
                if i_type in value:
                    res_conf = []
                    for _, configuration in value[i_type]:
                        res_conf = intersect_configs(res_conf, configuration, primary_bitwidth) if res_conf \
                            else configuration
                    if not res_conf:
                        raise Exception('Fake quantize node {} does not have a suitable configuration'
                                        ' for layers {}'.format(key, [layer for layer, _ in value[i_type]]))
                    conf[i_type] = _apply_preset_rule(preset_, key, i_type, res_conf, fqs_to_unify_)
            res[key] = conf
        res = _unify_and_apply_preset(preset_, res, fqs_to_unify_)
        return res

    available_presets = ['accuracy', 'mixed', 'performance']
    preset = config.preset
    if preset not in available_presets:
        raise Exception('Unsupported preset value: {}.'
                        ' Supported values are {}'.format(preset, available_presets))

    fqs_to_unify = find_fqs_to_unify(model, config)
    result = _intersect_and_apply_preset(preset, fq_to_hw_confs, fqs_to_unify)

    return result


def get_configurations_by_qscheme(fq_to_hw_confs, qscheme):
    """ Choose fake quantize configuration by qscheme
    :param fq_to_hw_confs: dictionary with fake quantize names as keys and
     list of its configurations as values (read_all_fake_quantize_configurations(..) return value)
    :param qscheme: The quantization scheme generated from the space
    :return dictionary with fake quantize nodes names as keys and
     suitable configuration chose by preset as values"""

    def _set_config(conf_by_layer, fq_type_):
        out = {}
        for node_name, _ in conf_by_layer:
            qscheme[node_name]['quantize'] = 1
        (node_name, _) = conf_by_layer[0]
        if qscheme[node_name]:
            out = qscheme[node_name][fq_type_]
        return out

    res = {}
    for key, value in fq_to_hw_confs.items():
        # fake quantize node can only have one type, so value dictionary will always have 1 element
        fq_type, confs = list(value.items())[0]
        res[key] = {fq_type: _set_config(confs, fq_type)}
    return res


def unify_recurrent_fqs(model, recurrent_hw_ops):
    recurrent_fqs_to_unify = []

    def source_fn(op):
        return [p for p in get_node_inputs(op) if p and p.type != 'Const']

    def criteria_fn(op):
        return op.type == 'FakeQuantize'

    def get_fqs_fullname(node, port):
        input_node = get_node_input(node, port)
        if criteria_fn(input_node):
            return [input_node.fullname]
        _, criteria = traverse_graph(input_node,
                                     move_fn=source_fn,
                                     stop_criteria_fn=criteria_fn,
                                     criteria_fns=criteria_fn)
        if criteria_fn in criteria:
            input_fqs = [reset_node_fullname(input_node.fullname, node_name) for node_name in criteria[criteria_fn]]
            return input_fqs
        return []

    def recurrent_fq_to_unify(cell_fullname, fqs):
        unique_fqs = set().union(*fqs)
        is_unified = all([get_node_input(get_node_by_name(model, name), 0).type != 'Const' for name in unique_fqs])
        if len(unique_fqs) >= 2 and is_unified:
            recurrent_fqs_to_unify.append([[cell_fullname], list(unique_fqs)])

    nodes = get_nodes_by_type(model, types=recurrent_hw_ops.keys(), recursively=True)
    for node in nodes:
        unify_group_indices = recurrent_hw_ops[node.type]
        for indices in unify_group_indices:
            fqs = [get_fqs_fullname(node, i) for i in indices]
            recurrent_fq_to_unify(node.fullname, fqs)

    return recurrent_fqs_to_unify


def find_fqs_to_unify(model, config):
    def _get_unified_scales_ops(hw_ops_):
        unified_scales_ops_ = []
        for hw_op in hw_ops_:
            if 'attributes' in hw_op and 'unified_scales' in hw_op['attributes'] and \
                hw_op['attributes']['unified_scales'] == 'all':
                del hw_op['attributes']['unified_scales']
                if not hw_op['attributes']:
                    del hw_op['attributes']
                unified_scales_ops_.append(hw_op)
        return unified_scales_ops_

    def _get_quantizable_per_ch_ops(hw_cfg_):
        _quantizable_per_ch_ops = []
        for hw_layer_cfg in hw_cfg_:
            if 'quantization' in hw_layer_cfg and \
               all(config['granularity'] == 'perchannel'
                   for config in hw_layer_cfg['quantization']['activations']):
                _quantizable_per_ch_ops.append(hw_layer_cfg['type'])
        return _quantizable_per_ch_ops

    def _is_special_unify_conditions(node):
        check_map = {
            'Concat': _is_concat_unify_condition
        }
        if node.type in check_map:
            logger.debug('Checking {} node with {} type'.format(node.fullname, node.type))
            return check_map[node.type](node)
        return True

    def _is_concat_unify_condition(node):
        def _is_followed_by_concat_unify_outs_op(input_node):
            if _is_quantize_agnostic_op(input_node):
                concat_stack.extend(get_all_node_outputs(input_node))
            elif input_node.type in [n['type'] for n in CONCAT_UNIFY_OUTPUTS]:
                concat_stack.clear()
                logger.debug('Found %s %s as Concat %s output',
                             input_node.type, input_node.fullname, node.fullname)
                return True
            return False

        def _is_followed_by_quantizable_per_channel(input_node):
            if _is_quantize_agnostic_op(input_node):
                concat_stack.extend(get_all_node_outputs(input_node))
            else:
                hw_op_type = get_hardware_config_operation_type(input_node, [hw_op['type'] for hw_op in hw_ops])
                if hw_op_type in per_channel_quantizable:
                    concat_stack.clear()
                    logger.debug('Found per channel quantizible op %s %s as Concat %s output',
                                 input_node.type, input_node.fullname, node.fullname)
                    return True
            return False

        def _source_fn(op):
            return [p for p in get_node_inputs(op) if p and p.type != 'Const']

        def _criteria_fn(op):
            return op.type in [n['type'] for n in CONCAT_UNIFY_INPUTS]

        _, criteria = traverse_graph(node,
                                    move_fn=_source_fn,
                                    stop_criteria_fn=_criteria_fn,
                                    criteria_fns=_criteria_fn)
        if not criteria:
            return False

        concat_stack = [node]
        while concat_stack:
            node_to_check = concat_stack.pop()
            res = _is_followed_by_concat_unify_outs_op(node_to_check)
        if res:
            concat_stack = [node]
            while concat_stack:
                node_to_check = concat_stack.pop()
                res = not _is_followed_by_quantizable_per_channel(node_to_check)

        return res

    def _is_agnostic_branching_op(node_):
        return node_.type == 'Concat'

    def _is_quantize_agnostic_op(node_):
        return bool(find_operation_matches(quantize_agnostic_ops, node_))

    def _is_unified_scales_op(node_):
        if bool(find_operation_matches(unified_scales_ops, node_)):
            return _is_special_unify_conditions(node_)
        return False

    def _has_const_input(layer):
        return 'Const' in [parent.type for parent in get_node_inputs(layer) if parent]

    def _get_unified_recurrent_scales_ops(hw_ops_):
        unified_scales_ops_ = {}
        for hw_op in hw_ops_:
            if hw_op['type'] in TYPES_TO_QUANTIZABLE_PORTS and 'unified_scales' in hw_op['attributes']:
                unified_scales_ops_[hw_op['type']] = hw_op['attributes']['unified_scales']
        return unified_scales_ops_

    def _process_node(node_, stack_, visited_, to_unify_):
        visited_[node_.fullname] = True
        if _is_unified_scales_op(node_) or _is_agnostic_branching_op(node_):
            if not _has_const_input(node_):
                to_unify_[0].append(node_.fullname)
        elif node_.type == 'FakeQuantize' and get_node_input(node_, 0).type != 'Const':
            to_unify_[1].append(node_.fullname)
        # traverse down
        if node_.type == 'FakeQuantize' or _is_quantize_agnostic_op(node_):
            for child in get_all_node_outputs(node_):
                node_data_type = get_node_data_type(child)
                if not visited_[child.fullname] and is_data_type_quantizable(node_data_type) and \
                        (_is_quantize_agnostic_op(child) or _is_unified_scales_op(child)):
                    stack_.append(child)
        # traverse up
        if node_.type != 'FakeQuantize':
            for parent in get_node_inputs(node_):
                node_data_type = get_node_data_type(parent)
                if parent and not visited_[parent.fullname] and is_data_type_quantizable(node_data_type) and \
                        (parent.type == 'FakeQuantize' or _is_quantize_agnostic_op(parent)):
                    stack_.append(parent)

    hardware_config = load_hardware_config(config)
    per_channel_quantizable = _get_quantizable_per_ch_ops(hardware_config)
    hw_ops = get_operation_list(hardware_config)
    quantize_agnostic_ops = [op[1] for op in find_operation_matches(QUANTIZE_AGNOSTIC_OPERATIONS, hw_ops)]
    recurrent_hw_ops = _get_unified_recurrent_scales_ops(hw_ops)
    unified_scales_ops = _get_unified_scales_ops(hw_ops)
    if not (unified_scales_ops or recurrent_hw_ops):
        return []

    visited = defaultdict(lambda: False)
    fqs_to_unify = []
    if model is None:
        return fqs_to_unify
    for fq in get_nodes_by_type(model, ['FakeQuantize']):
        if not visited[fq.fullname] and get_node_input(fq, 0).type != 'Const':
            stack = [fq]
            to_unify = [[], []]
            while stack:
                node = stack.pop()
                _process_node(node, stack, visited, to_unify)

            if to_unify[0] and \
                    any([_is_unified_scales_op(get_node_by_name(model, bridge)) for bridge in to_unify[0]]) and \
                    len(to_unify[1]) > 1:
                fqs_to_unify.append(to_unify)

    recurrent_fqs = unify_recurrent_fqs(model, recurrent_hw_ops)
    fqs_to_unify.extend(recurrent_fqs)

    fqs_to_unify = sorted([[sorted(c[0]), sorted(c[1])] for c in fqs_to_unify])
    logger.debug('Operations and corresponding fake quantize nodes to unify scales:')
    for ops, fqs in fqs_to_unify:
        logger.debug('Operations: {}'.format(ops))
        logger.debug('Fake quantize nodes: {}'.format(fqs))
    logger.debug('')

    return fqs_to_unify


def _fake_quantize_to_types(model, hardware_config):
    """ Helper function to bypass graph and get fake quantize node
    children nodes with predefined types
    :return dictionary with fake quantize node name as a key and tuple with list of
    its quantizable descendant types and boolean specifying if fake quantize node is weights
    """

    def _is_quantizable(node):
        return not find_operation_matches(quantize_agnostic_ops, node)

    def _get_node_valuable_descendant(node):
        descendants = []
        queue = deque([node])
        while queue:
            current = queue.popleft()
            children = get_all_node_outputs(current)
            for child in sorted(children, key=lambda child: child.fullname):
                if not _is_quantizable(child):
                    queue.append(child)
                elif child.type not in descendants:
                    descendants.append((child.fullname,
                                        get_hardware_config_operation_type(child, available_types)))
                if current.type == 'Split' \
                        and child.type == 'Concat' \
                        and len({child_.fullname for child_ in children}) == 1:
                    break
        return descendants

    hw_ops = get_operation_list(hardware_config)
    quantize_agnostic_ops = [op[1] for op in
                             find_operation_matches(QUANTIZE_AGNOSTIC_OPERATIONS, hw_ops)]

    out = {}
    available_types = [layer['type'] for layer in hardware_config]
    for fq in get_nodes_by_type(model, ['FakeQuantize']):
        if fq['fq_group'] == 'outputs':
            fq_input = get_node_input(fq, 0)
            hw_node_types = get_hardware_config_operation_type(fq_input, available_types)
            out_data = ([(fq_input.name, hw_node_types)], fq['fq_group'])
        else:
            out_data = (_get_node_valuable_descendant(fq), fq['fq_group'])
        out[fq.fullname] = out_data

    return out


def change_configurations_by_model_type(model, config, fq_configuration, hardware_config):
    if config['model_type'] == 'transformer' and config['target_device'] in ['ANY', 'CPU', 'CPU_SPR', 'GPU', 'VPU']:
        change_configurations_by_model_type_transformer(model, fq_configuration, hardware_config)


def change_configurations_by_model_type_transformer(model, fq_configuration, hardware_config):
    fq_types = _fake_quantize_to_types(model, hardware_config)
    for fq in get_nodes_by_type(model, ['FakeQuantize']):
        node_creator_fq, fq_group = fq_types[fq.fullname]
        is_weights = fq_group == 'weights'
        node_name = None
        for name, type_node in node_creator_fq:
            if type_node == 'MatMul':
                node_name = name

        if node_name is None or is_weights:
            continue

        node = get_node_by_name(model, node_name)

        if not check_const_input(node):
            fq_configuration[fq.name]['activations'] = deepcopy(fq_configuration[fq.name]['activations'])
            fq_configuration[fq.name]['activations']['mode'] = 'symmetric'
