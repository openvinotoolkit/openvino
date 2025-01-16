# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ast
import logging as log
import re
from collections import defaultdict
from copy import copy

import numpy as np
from openvino.runtime import PartialShape, Dimension

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension_value, shape_array
from openvino.tools.mo.front.onnx.extractors.utils import get_backend_pad
from openvino.tools.mo.graph.graph import Node, Graph, add_opoutput
from openvino.tools.mo.middle.passes.eliminate import reverse_dfs
from openvino.tools.mo.utils import class_registration
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.unsupported_ops import UnsupportedOps
from openvino.tools.mo.utils.utils import refer_to_faq_msg


def restore_edges(graph: Graph, get_edges: callable):
    """
    Take a graph without edges and extract dependencies between nodes with the help of get_edges function.
    For a given node n the get_edges function returns a list of tuples (n1, n2, attrs), that is used to create
    n1 --> n2 edge with attributes attrs.
    It is possible that two nodes n1 and n2 have more than one n1 --> n2 edges, so the resulting graph is Graph.
    """
    used_tensors = set()
    for node in list(graph.nodes()):
        edges = get_edges(Node(graph, node))
        for u, v, d in edges:
            undefined = ['"' + x + '"' for x in [u, v] if not graph.has_node(x)]
            if len(undefined):
                raise Error(
                    'While creating an edge from "{}" to "{}": node name {} is undefined in the graph. ' +
                    'Check correctness of the input model. ',
                    u, v,
                    ' and '.join(undefined) +
                    refer_to_faq_msg(25)
                )
            used_tensors.add(u)

        graph.add_edges_from(edges)
    return used_tensors


def remove_control_dependency_inputs(graph: Graph):
    """
    Delete control dependency inputs from pb all over the graph
    :param graph: graph to operate on 
    """
    for _, attrs in list(graph.nodes(data=True)):
        if 'pb' not in attrs:
            continue
        pb = attrs['pb']
        ind = 0
        while ind < len(pb.input):
            if pb.input[ind].startswith('^'):
                del pb.input[ind]
            else:
                ind += 1


def update_attrs(attrs: [dict, Node], attr: str, new: [str, list]):
    """ Updates attrs[attr], which should be a list, by a new items from 'new' list.
    If attrs[attr] doesn't exist, create it.
    """
    if attr not in attrs:
        attrs[attr] = []
    if isinstance(new, str):
        new = [new]
    attrs[attr] = list(set(attrs[attr]).union(set(new)))


def add_attrs_props(attrs: dict):
    update_attrs(attrs, 'dim_attrs', ['spatial_dims', 'channel_dims', 'batch_dims', 'axis'])
    update_attrs(attrs, 'shape_attrs', ['shape', 'pad', 'window', 'stride', 'output_shape'])
    return attrs


def spatial_attr_getter(node: Node, field: str = None, dim: int = None, post: callable = None):
    """

    Parameters
    ----------
    node: node of graph
    field: name of the field in original layer
    dim: dimension of the field
    post: function for getting values of the field

    Returns:
        value of field
    -------

    """
    if node.has(field) and type(node[field]) is np.ndarray and node.has('spatial_dims'):
        return post(node[field][node.spatial_dims[dim]])
    return None


def spatial_getter(name: str, field: str, dim: int, post: callable = lambda x: x):
    """

    Parameters
    ----------
    name: name of the filed in IR
    field: name of field in original layer
    dim: dimension of field
    post: function for getting values of field

    Returns:
        of the filed in IR  and function for getting values of the field
    -------

    """
    return name, lambda node: spatial_attr_getter(node, field=field, dim=dim, post=post)


def attr_getter(node: Node, name: str):
    if node.has(name):
        if type(node[name]) is list or type(node[name]) is np.ndarray:
            return ','.join(map(str, node[name]))
        elif type(node[name]) is not np.ndarray:
            return str(node[name])
    return None


def bool_to_str(node: Node, attr: str):
    # Function converts 0/1 or bool False/True or '0'/'1' values to str 'false'/'true' which need to appear in IR
    attribute_name = node.soft_get(attr, None)
    if attribute_name is None:
        return None
    if isinstance(attribute_name, bool):
        return str(attribute_name).lower()
    elif attribute_name in [0, 1]:
        return str(bool(attribute_name)).lower()
    elif attribute_name in ['0', '1']:
        return str(bool(int(attribute_name))).lower()
    else:
        raise Error('Wrong value {} for boolean attribute {} in node {}'.format(
            attribute_name, attr, node.soft_get('name')))


def kernel_getter(node: Node, dim: int):
    if node.kind == 'op' and node.op in ['Conv2D', 'DepthwiseConv2dNative', 'Deconv2D']:
        if node.has('kernel_spatial'):
            return node.kernel_spatial[dim]  # TODO check if order of dimension matches expectations
        weights = node.in_node(1)  # WARNING: 1 is hardcoded input with a kernel
        return weights.shape[weights.spatial_dims[dim]]
    else:
        return None


def node_defs_to_str(node: Node):
    node_name_to_pb_mapping = {node_name: node_def for node_name, node_def in node['pbs'].items()}
    result = ''
    for node_name in node['nodes_order']:
        result += 'node {\n' + str(node_name_to_pb_mapping[node_name]) + '}\n'
    return result


def update_ie_fields(attrs: dict, ir_version = None):
    ir_v10_attrs = {
        'IE': [(
            'layer',
            [('id', lambda node: node.node), 'name', 'type', 'version'],
            [
                (
                    'data',
                    [
                        'auto_pad',
                        'epsilon',
                        'min',
                        'max',
                        ('axis', lambda node: attr_getter(node, 'axis')),
                        'tiles',
                        ('dim', lambda node: attr_getter(node, 'dim')),
                        'num_axes',
                        ('pool-method', 'pool_method'),
                        'group',
                        ('rounding-type', 'rounding_type'),
                        ('exclude-pad', 'exclude_pad'),
                        'operation',
                        'out-size',
                        'power',
                        'shift',
                        'alpha',
                        'beta',
                        'coords',
                        'classes',
                        'num',
                        ('local-size', 'local_size'),
                        'region',
                        'knorm',
                        'bias',

                        'num_classes',
                        'keep_top_k',
                        'variance_encoded_in_target',
                        'code_type',
                        'share_location',
                        'nms_threshold',
                        'confidence_threshold',
                        'background_label_id',
                        'top_k',
                        'eta',
                        'visualize',
                        'visualize_threshold',
                        'save_file',
                        'output_directory',
                        'output_name_prefix',
                        'output_format',
                        'label_map_file',
                        'name_size_file',
                        'num_test_image',
                        'prob',
                        'resize_mode',
                        'height',
                        'width',
                        'height_scale',
                        'width_scale',
                        'pad_mode',
                        'pad_value',
                        'interp_mode',

                        'img_size',
                        'img_h',
                        'img_w',
                        'step',
                        'step_h',
                        'step_w',
                        ('offset', lambda node: attr_getter(node, 'offset')),
                        'variance',
                        'flip',
                        'clip',
                        ('min_size', lambda node: attr_getter(node, 'min_size')),
                        ('max_size', lambda node: attr_getter(node, 'max_size')),
                        ('aspect_ratio', lambda node: attr_getter(node, 'aspect_ratio')),
                        'decrease_label_id',
                        'normalized',
                        'scale_all_sizes',

                        ('type', 'norm_type'),
                        'eps',
                        'eps_mode',
                        'across_spatial',
                        'channel_shared',

                        'negative_slope',
                        'engine',

                        'num_filter',
                        ('type', 'sample_type'),
                        ('order', lambda node: attr_getter(node, 'order')),

                        'pooled_h',
                        'pooled_w',
                        'spatial_scale',

                        'cls_threshold',
                        'max_num_proposals',
                        'iou_threshold',
                        'min_bbox_size',
                        'feat_stride',
                        'pre_nms_topn',
                        'post_nms_topn',
                        ('type', lambda node: node['filler_type'] if node.has('filler_type') else None),
                        ('value', lambda node: node['filler_value'] if node.has('filler_value') else None),
                        ('output',
                         lambda node: node.output_shape[node.channel_dims][0] if node.has('output_shape') and node.has(
                             'channel_dims') else None),
                        ('input_nodes_names', lambda node: ' '.join(node['input_nodes_names']) if node.has(
                            'input_nodes_names') else None),
                        ('output_tensors_names', lambda node: ' '.join(node['output_tensors_names']) if node.has(
                            'output_tensors_names') else None),
                        ('real_input_dims', lambda node: ';'.join([' '.join(map(str, shape)) for shape in
                                                                   node['real_input_dims']])
                        if node.has('real_input_dims') else None),
                        ('protobuf', lambda node: node_defs_to_str(node) if node.has('pbs') else None),
                        {'custom_attributes': None},
                        ('strides', lambda node: ','.join(map(str, node['stride'][node.spatial_dims])) if node.has_valid('stride') else None),
                        ('kernel', lambda node: ','.join(map(str, node['kernel_spatial'])) if node.has_valid(
                            'kernel_spatial') else None),
                        ('dilations', lambda node: ','.join(map(str, node['dilation'][node.spatial_dims])) if node.has_valid('dilation') else None),

                        ('pads_begin', lambda node: ','.join(map(str, get_backend_pad(node.pad, node.spatial_dims, 0))) if node.has_valid('pad') else None),
                        ('pads_end', lambda node: ','.join(map(str, get_backend_pad(node.pad, node.spatial_dims, 1))) if node.has_valid('pad') else None),

                        ('scale', lambda node: attr_getter(node, 'scale')),
                        'crop_width',
                        'crop_height',
                        'write_augmented',
                        'max_multiplier',
                        'augment_during_test',
                        'recompute_mean',
                        'write_mean',
                        'mean_per_pixel',
                        'mode',
                        'bottomwidth',
                        'bottomheight',
                        'chromatic_eigvec',
                        'kernel_size',
                        'max_displacement',
                        'stride_1',
                        'stride_2',
                        'single_direction',
                        'do_abs',
                        'correlation_type',
                        'antialias',
                        'resample_type',
                        'factor',
                        'coeff',
                        ('ratio', lambda node: attr_getter(node, 'ratio')),
                        'size',
                    ],
                    []),
                '@ports',
                '@consts'])]
    }

    ir_version_mapping = {
        # Default behaviour is IR V10 attributes
        None: ir_v10_attrs,
        10: ir_v10_attrs,
        11: ir_v10_attrs,
    }

    if ir_version not in ir_version_mapping.keys():
        raise Error("Unrecognized IR version was specified: {}".format(ir_version))

    attrs.update(ir_version_mapping[ir_version])


def create_tensor_nodes(graph: Graph):
    """
    Creates nodes between ops to represent intermediate data that flows from one op to another.
    For each edge with unique out attribute that goes from a given node,
    a new node is created with attribute kind='data'

        Old: op1 ---(out, in)---> op2
        New: op1 ---(out)---> tensor ---(in)---> op2

    Edge also can contain in_attrs, out_attrs and data_attrs attributes. Each of them is a list
    of names of other attributes in an edge. The lists control how original edge attributes are distributed
    among newly created in/out edges and tensor node. Having the name X in in_attrs means that an edge attribute
    with name X should be moved to the input edge to the tensor (together with 'out' attribute).
    Having Y in out_attrs means that the attribute Y should be moved to the output edge from the tensor.
    And if Z is in data_attrs, Z attribute of the edge should be moved to the tensor node itself.
    For example:

        Old: op1 ---(out, in, X, Y, Z)---> op2
        New: op1 ---(out, X)---> tensor(Z) ---(in, Y)---> op2

    All old nodes are marked as kind='op'
    """
    for node in list(graph.nodes()):
        node_attr = graph.node[node]
        # threat all existing nodes in the graph as operation nodes (in opposite to data nodes created in this function
        # below)
        graph.node[node]['kind'] = 'op'

        # the Result nodes are just marker operations so we don't need to create output tensors for them
        if node_attr['op'] == 'Result':
            continue
        # out_edges is a list of (u, v, d), where d is a dict of edge attributes
        out_edges = list(graph.out_edges(node, data=True))

        # Make a list of unique output ports for a node, unique means an edge has unique 'out' attribute.
        # Multiple edges coming from node may have duplicated 'out' ports because a single output port
        # can be reused multiple times by several consumers.
        out_ports = list(set([d['out'] for u, v, d in out_edges]))

        smart_node = Node(graph, node)
        out_nodes = smart_node.out_nodes()
        node_name = str(smart_node.name) if smart_node.has_valid('name') else str(smart_node.id)

        # assign to each output port a tensor unique id in the graph
        out_tensor_dict = {port: graph.unique_id('{}/Output_{}/Data_'.format(node_name, port)) for port in out_ports}

        # add a new node with kind='data' per each tensor
        graph.add_nodes_from([(uid,
                               add_attrs_props(
                                   dict(name=uid, kind='data', shape=None, value=None, data_type=None, infer=None))) for
                              port, uid in out_tensor_dict.items()])

        # add all edges from the node to each output port tensor
        added_out_ports = set()

        for src_node, _, attrs in out_edges:
            port = attrs['out']
            if port not in added_out_ports:
                graph.add_edges_from([(node, out_tensor_dict[port], get_specific_edge_attrs(attrs, 'out_attrs'))])
                # merge additional data node attributes from original edge
                graph.node[out_tensor_dict[port]].update(get_specific_edge_attrs(attrs, 'data_attrs'))
                added_out_ports.add(port)
        # graph.add_edges_from([(node, out_tensor_dict[port], {'out' : port}) for port in out_ports])

        # connect newly created tensor nodes to their consumers
        for u, v, d in out_edges:
            graph.add_edges_from([(out_tensor_dict[d['out']], v, get_specific_edge_attrs(d, 'in_attrs'))])
        # graph.add_edges_from([(out_tensor_dict[d['out']], v, {'in' : d['in']}) for u, v, d in out_edges])
        # remove old edges op1 ---> op2; due to bug in nx, need to repack out_edges to have (u,v) as an element
        graph.remove_edges_from([x[:2] for x in out_edges])
    return graph


# 'attrs_type' is either "in_attrs" or "out_attrs"
# update result values with the values from dictionary additional_attrs
def get_specific_edge_attrs(attrs: dict, attrs_type: str, additional_attrs=None):
    new_attrs = dict()
    if attrs_type in attrs:
        for key in attrs[attrs_type]:
            if key in attrs.keys():
                new_attrs[key] = attrs[key]
    if additional_attrs is not None:
        new_attrs.update(additional_attrs)
    return new_attrs


def extract_node_attrs(graph: Graph, extractor: callable):
    """
    For each node produce new entries in a node attributes dictionary by existing attributes.
    Old attributes are not removed but merged with new ones.
    """
    unsupported = UnsupportedOps(graph)
    for node, attrs in list(graph.nodes(data=True)):
        # the 'Result' operation is a virtual operation that is added after the output nodes
        if 'op' in attrs and attrs['op'] == 'Result':
            supported, new_attrs = True, {'in_attrs': list(), 'out_attrs': list()}
        else:
            try:
                supported, new_attrs = extractor(Node(graph, node))
            except Exception as e:
                log.warning('Node attributes: {}'.format(graph.node[node]))
                raise Error(
                    'Unexpected exception happened during extracting attributes for node {}.' +
                    '\nOriginal exception message: {}',
                    node,
                    str(e)
                ) from e
        if supported:
            if 'IE' not in new_attrs:
                update_ie_fields(new_attrs)
            add_attrs_props(new_attrs)
        for key, val in new_attrs.items():
            graph.node[node][key] = val
        if not supported:
            unsupported.add(Node(graph, node))

    unsupported.report(log.warning, 'Instructions/layers that do not have attribute extractors:')

    return graph


def raise_no_node(node_name: str):
    raise Error('No node with name {}'.format(node_name))


def raise_node_name_collision(node_name: str, found_nodes: list):
    raise Error('Name collision was found, there are several nodes for mask "{}": {}. '
                'If your intention was to specify port for node, please instead specify node names connected to '
                'this port. If your intention was to specify the node name, please add port to the node '
                'name'.format(node_name, found_nodes))


def get_node_id_with_ports(graph: Graph, node_name: str, skip_if_no_port=True):
    """
    Extracts port and node ID out of user provided name
    :param graph: graph to operate on
    :param node_name: user provided node name
    :return: node ID, direction of port ('in', 'out', 'port') and port number or None
    """
    node_names = [n.soft_get('name', n.id) for n in graph.get_op_nodes()]
    found_names = []
    for name in node_names:
        regexp = r'(\d*:)?(' + name + r')(:\d*)?'
        match = re.search(regexp, node_name)
        if match and match.group() == node_name:
            in_port = None
            out_port = None
            if match.group(1) and match.group(3):
                log.warning('Skipping the case with both in and out port specified, only one port can be specified')
                continue
            node = Node(graph, graph.get_node_id_by_name(name))
            if match.group(1):
                in_port = int(match.group(1).replace(':', ''))
                if skip_if_no_port and in_port not in [e['in'] for e in node.in_edges().values()]:
                    # skip found node if it doesn't have such port number
                    continue
            if match.group(3):
                out_port = int(match.group(3).replace(':', ''))
                if skip_if_no_port and out_port not in [e['out'] for e in node.out_edges().values()]:
                    # skip found node if it doesn't have such port number
                    continue

            found_names.append((in_port, out_port, name))
    if len(found_names) == 0:
        raise_no_node(node_name)
    if len(found_names) > 1:
        raise_node_name_collision(node_name, [name for _, _, name in found_names])
    in_port, out_port, name = found_names[0]
    node_id = graph.get_node_id_by_name(name)
    if in_port is not None:
        direction = 'in'
        port = in_port
    elif out_port is not None:
        direction = 'out'
        port = out_port
    else:
        direction = 'port'
        port = None
    return node_id, direction, port


def get_new_placeholder_name(node_id: str, is_out_port: bool = False, port: int = 0):
    """
    Forms a name of new placeholder created by cutting a graph
    :param node_id: a node name that is cut
    :param is_out_port: it is True iff output port is cut
    :param port: a port number
    :return: a name of new placeholder created by cutting a graph
    """
    port_type = '_out' if is_out_port else ''
    return '{}/placeholder{}_port_{}'.format(node_id, port_type, port)


def create_params_with_custom_types(packed_user_shapes: [None, dict]):
    """
    Compute a list of placeholder names for which an user specifies custom type
    :param packed_user_shapes: packed data that contains input node names,
    their port numbers, shapes and data types
    :return: a list of placeholder names for which an user specifies custom type
    Example of packed_user_shapes dictionary:
    packed_user_shapes =
    {
        'node_ID':
            [
                {'shape': None, 'in': 0},
                {'shape': None, 'in': 1},
            ],
        'node_1_ID':
            [
                {'shape': [1, 227, 227, 3], 'port': None, 'data_type': np.int32}
            ],
        'node_2_ID':
            [
                {'shape': None, 'out': 3}
            ]
    }
    For which the function returns a list ['node_1_ID'] because this node only has custom data type
    """
    if packed_user_shapes is None:
        return []

    params_with_custom_types = []
    for input_name in packed_user_shapes:
        for desc in packed_user_shapes[input_name]:
            p_name = input_name
            if 'port' in desc and desc['port'] is None:  # neither input nor output port specified
                user_defined_type = desc.get('data_type', None)
            else:  # need to check the particular port the Parameter was created for
                p_name = get_new_placeholder_name(input_name, 'out' in desc,
                                                  desc['out'] if 'out' in desc else desc['in'])
                user_defined_type = desc.get('data_type', None)
            if user_defined_type is not None:
                params_with_custom_types.append(p_name)
    return params_with_custom_types


def input_user_data_repack(graph: Graph, input_user_shapes: [None, list, dict, np.ndarray],
                           freeze_placeholder: dict, input_user_data_types = dict()):
    """
    Restructures user input cutting request. Splits ports out of node names. Transforms node names to node ids.
    :param graph: graph to operate on
    :param input_user_shapes: data structure representing user input cutting request. It may be:
    # None value if user did not provide neither --input nor --input_shape keys
    # list instance witch contains input layer names with or without ports if user provided only --input key
    # dict instance witch contains input layer names with or without ports as keys and shapes as values if user
        provided both --input and --input_shape
    # np.ndarray if user provided only --input_shape key
    :param freeze_placeholder: dictionary with placeholder names as keys and freezing value as values
    :param input_user_data_types: dictionary with input nodes and its data types
    :return: restructured input shapes and freeze placeholder shapes information
    Example of input dictionary:
    _input_shapes =
    {
        'node_ID':
            [
                {'shape': None, 'in': 0},
                {'shape': None, 'in': 1},
            ],
        'node_1_ID':
            [
                {'shape': [1, 227, 227, 3], 'port': None, 'data_type': np.int32}
            ],
        'node_2_ID':
            [
                {'shape': None, 'out': 3}
            ]
    }
     Example of freeze placeholder dictionary:
    _freeze_placeholder =
    {
        'phase_train' : False
    }
    """
    _input_shapes = defaultdict(list)
    _freeze_placeholder = dict()
    _freeze_new_placeholder = defaultdict(list)

    # freeze placeholder restructure
    # Replaces placeholder name with placeholder id. Raises if there is no placeholder with such ID
    placeholders_ids = graph.get_nodes_with_attributes(op='Parameter')
    if freeze_placeholder is None:
        _freeze_placeholder = None
    else:
        if isinstance(freeze_placeholder, list):
            raise Error('Unnamed inputs with values are not supported for legacy frontend. Please provide input names.')
        for placeholder_name, value in freeze_placeholder.items():
            placeholder_id, direction, port = get_node_id_with_ports(graph, placeholder_name)
            if port is None and placeholder_id in placeholders_ids:
                _freeze_placeholder[placeholder_id] = value
            else:
                # collect possible new placeholders that will be frozen with values
                is_out_port = (direction == 'out')
                new_placeholder_id = get_new_placeholder_name(placeholder_id, is_out_port, port)
                _freeze_new_placeholder[placeholder_id].append(
                    {'direction': direction, 'port': port, 'name': placeholder_name, 'id': new_placeholder_id,
                     'value': value})

    if isinstance(input_user_shapes, list):
        if len(input_user_shapes) == 1 and isinstance(input_user_shapes[0], PartialShape):
            input_user_shapes = input_user_shapes[0]

    # input user shapes restructure
    if input_user_shapes is None:
        # None User did not provide neither --input nor --input_shape keys
        _input_shapes = None
    elif isinstance(input_user_shapes, list) and len(input_user_shapes) > 1 and isinstance(input_user_shapes[0], PartialShape):
        raise Error('Please provide input layer names for input layer shapes. ' + refer_to_faq_msg(58))
    elif isinstance(input_user_shapes, list) or isinstance(input_user_shapes, dict):
        # list [layer names w or w/o ports]. User provided only --input key
        # dict {layer names w or w/o ports as keys: shapes as values}. User provided both --input and --input_shape
        for input_name in input_user_shapes:
            node_id, direction, port = get_node_id_with_ports(graph, input_name)
            shape = None if isinstance(input_user_shapes, list) else input_user_shapes[input_name]
            if input_name in input_user_data_types and input_user_data_types[input_name] is not None:
                data_type = input_user_data_types[input_name]
                _input_shapes[node_id].append({'shape': shape, direction: port, 'data_type': data_type})
            else:
                _input_shapes[node_id].append({'shape': shape, direction: port})
        if _freeze_placeholder is not None:
            # here we give user an opportunity not to provide node names from --freeze_placeholder_with_value in --input
            [_input_shapes[ph_id].append({'shape': None, 'port': None}) for ph_id in _freeze_placeholder
             if ph_id not in _input_shapes]
    else:
        # User provided only --input_shape key
        assert isinstance(input_user_shapes, PartialShape)
        if len(placeholders_ids) == 1:
            # There is only one placeholder in the original network
            _input_shapes[placeholders_ids[0]].append({'shape': input_user_shapes, 'port': None})
        elif _freeze_placeholder is not None:
            # There are multiple placeholders and some of them are frozen
            original_phs = copy(placeholders_ids)
            [placeholders_ids.remove(ph_id) for ph_id in _freeze_placeholder]
            if len(placeholders_ids) != 1:
                raise Error('Original placeholders: \'{}\'. Freezing was requested for \'{}\'. --input_shape was '
                            'provided without --input. Can not deduce which node shape to override'
                            ''.format(', '.join(original_phs), ', '.join(_freeze_placeholder.keys())))
            _input_shapes[placeholders_ids[0]].append({'shape': input_user_shapes, 'port': None})
            [_input_shapes[node_id].append({'shape': None, 'port': None}) for node_id in _freeze_placeholder]
        else:
            # There are multiple placeholders in the original network and none of them are frozen
            # Can not deduce which placeholder shape to override
            raise Error('No or multiple placeholders in the model, but only one shape is provided, cannot set it. ' +
                        refer_to_faq_msg(32))

    # check that shape is specified for every new placeholder in _input_shapes
    # and update _freeze_placeholder with new possible placeholders created by cutting a graph
    for node_id in _freeze_new_placeholder:
        new_phs = _freeze_new_placeholder[node_id]
        if node_id not in _input_shapes:
            raise Error('Shape is not specified for the placeholder with name {} through --input_shape option.'
                        ''.format(new_phs[0]['name']))
        _ins = _input_shapes[node_id] # list
        for new_ph in new_phs:
            name = new_ph['name']
            direction = new_ph['direction']
            port = new_ph['port']
            placeholder_id = new_ph['id']
            value = new_ph['value']
            if any([_in['shape'] is not None and direction in _in and _in[direction] == port for _in in _ins]):
                _freeze_placeholder[placeholder_id] = value
            else:
                raise Error('Shape is not specified for the placeholder with name {} through --input_shape option.'
                            ''.format(name))

    return _input_shapes, _freeze_placeholder


def output_user_data_repack(graph: Graph, outputs: list):
    """

    :param graph: graph to operate on
    :param outputs: list of node names provided by user
    :return: dictionary with node IDs as keys and list of port dictionaries as values
    Example of outputs dictionary:
    _outputs =
    {
        'node_ID':
            [
                {'out': 0},
                {'out': 1},
            ],
        'node_1_ID':
            [
                {'port': None}
            ],
        'node_2_ID':
            [
                {'in': 3}
            ]
    }
    """
    _outputs = defaultdict(list)
    if outputs is None:
        _outputs = None
    else:
        for output in outputs:
            node_id, direction, port = get_node_id_with_ports(graph, output)
            _outputs[node_id].append({direction: port})
    return _outputs


def user_data_repack(graph: Graph, input_user_shapes: [None, list, dict, np.array],
                     input_user_data_types: dict, outputs: list, freeze_placeholder: dict):
    """
    :param graph: graph to operate on
    :param input_user_shapes: data structure representing user input cutting request
    :param outputs: list of node names to treat as outputs
    :param freeze_placeholder: dictionary with placeholder names as keys and freezing value as values
    :return: restructured input, output and freeze placeholder dictionaries or None values
    """
    _input_shapes, _freeze_placeholder = input_user_data_repack(graph, input_user_shapes, freeze_placeholder,
                                                                input_user_data_types=input_user_data_types)
    _outputs = output_user_data_repack(graph, outputs)
    return _input_shapes, _outputs, _freeze_placeholder


def add_output_ops(graph: Graph, user_defined_outputs: dict, inputs: dict = None):
    sinks = []
    # func sets all layers as outputs in case of empty user_defined_outputs list (it's impossible to reach by cli)
    assert not (isinstance(user_defined_outputs, list) and not len(user_defined_outputs))

    # remove previously generated Result if any
    graph.remove_nodes_from([node_name for node_name in graph.nodes() if
                             'op' in graph.node[node_name] and graph.node[node_name]['op'] == 'Result'])

    if user_defined_outputs is None:
        inputs = graph.get_nodes_with_attributes(op='Parameter') if inputs is None else list(inputs.keys())
        input_reachable, dead_outputs, undead_outputs = set(), [], []
        for input in inputs:
            graph.dfs(node_name=input, visited=input_reachable)
        for node_name in list(graph.nodes()):
            if len(list(graph.out_edges(node_name))) == 0:
                if node_name in input_reachable:
                    out_ports_count = Node(graph, node_name).out_ports_count if Node(graph, node_name).has_valid('out_ports_count') else 1
                    for i in range(out_ports_count):
                        sinks.append(add_opoutput(graph, node_name, i, False))
                    undead_outputs.append(node_name)
                else:
                    dead_outputs.append(node_name)
        if len(dead_outputs):
            log.info('Possible outputs: \'{!s}\' are not input reachable. True outputs are {!s}'
                     ''.format(', '.join([str(d_o) for d_o in dead_outputs]),
                               ', '.join([str(u_o) for u_o in undead_outputs])))
    else:   # cutting the net by outputs
        for node, values in user_defined_outputs.items():
            if node not in graph.nodes():
                raise Error('Node "{}" does not exist in the graph. ' +
                            refer_to_faq_msg(26), node)
            for value in values:
                if 'in' in value:
                    in_edges = list(graph.in_edges(node, data=True))
                    if len(in_edges) - 1 < value['in']:
                        raise Error('Port index {} is out of number of available input ports for node "{}". ' +
                                    refer_to_faq_msg(29), value['in'], node)
                    for u, v, attrs in in_edges:
                        if 'in' in attrs and attrs['in'] == value['in']:
                            sinks.append(add_opoutput(graph, u, attrs['out'], user_defined_name=node))
                elif 'out' in value:
                    out_edges = list(graph.out_edges(node, data=True))
                    if len(out_edges) - 1 < value['out']:
                        raise Error('Port index {} is out of number of available output ports for node "{}". ' +
                                    refer_to_faq_msg(29), value['out'], node)
                    for u, v, attrs in out_edges:
                        if 'out' in attrs and attrs['out'] == value['out']:
                            sinks.append(add_opoutput(graph, node, attrs['out'], user_defined_name=node))
                else:
                    sinks.append(add_opoutput(graph, node, 0, user_defined_name=node))
    return sinks


def add_outputs_identity(graph: Graph, outputs: list, add_edge: callable, params: dict = {}):
    """
    Adds identity nodes marked with needs_removal=True attribute after each output of the graph.
    These nodes are used for storing tensor names information at the incoming edge
    and are removed with the OutputCut transformation.
    :param graph: graph to operate on.
    :param outputs: list of output node ids.
    :param add_edge: method which adds an edge to the graph with the following signature:
     f(src_node_id: str, dst_node_id: str, in_port: int).
    :param params: extra parameters for add_edge method.
    """
    for output in outputs:
        fake_node_name = graph.unique_id(output)
        graph.add_node(fake_node_name, name=fake_node_name, identity=True, kind='op', op='Identity',
                       infer=None, needs_removal=True, symbol_dict={'op': 'Identity'})
        add_edge(graph, output, fake_node_name, **params)


def set_is_input(graph: Graph, placeholders: list, is_input: bool):
    for placeholder in placeholders:
        graph.node[placeholder]['is_input'] = is_input


def check_input(graph: Graph, node_name: str):
    node = Node(graph, node_name)
    if node['kind'] == 'op' and node['op'] == 'Parameter' and not len(graph.in_edges(node_name)) and \
            not node['is_input']:
        raise Error("--input parameter was provided. Other inputs are needed for output computation. "
                    "Provide more inputs or choose another place to cut the net. " + refer_to_faq_msg(27))


def split_node_in_port(node_id: str):
    """Split node_id in form port:node to separate node and port, where port is converted to int"""
    if isinstance(node_id, str):
        separator = ':'
        parts = node_id.split(separator)
        if len(parts) > 1:
            if parts[0].isdigit():
                node_name = separator.join(parts[1:])
                try:
                    port = int(parts[0])
                    return node_name, port
                except ValueError as err:
                    log.warning('Didn\'t recognize port:node format for "{}" because port is not an integer.'.format(
                    node_id))
            else:
                node_name = separator.join(parts[:-1])
                try:
                    port = int(parts[-1])
                    return node_name, port
                except ValueError as err:
                    log.warning('Didn\'t recognize node:port format for "{}" because port is not an integer.'.format(
                    node_id))

    return node_id, None


def add_input_op_input_port_without_data(graph: Graph, node_id: str, input_op, edge_attrs: dict):
    input_node = input_op.create_node()
    graph.add_edge(input_node.id, node_id, **edge_attrs)
    log.debug('Input: {} for node {}'.format(input_node.id, node_id))
    log.debug("Add edge from {} to {}".format(input_node.id, node_id))
    return input_node.id


def add_input_op_input_port_with_data(graph: Graph, node_id: str, input_op, edge_attrs: dict):
    assert graph.stage == 'middle', 'add_input_op_input_port_with_data() function can be used only for graph after ' \
                                    'shape inference!'
    input_node = input_op.create_node(edge_attrs=edge_attrs)
    node = Node(graph, node_id)

    out_port = input_node.out_port(edge_attrs['out'])
    out_port.connect(node.in_port(edge_attrs['in']))
    out_port.data.set_shape(input_node.soft_get('shape', None))
    input_data_node = input_node.out_node(0)

    if 'fw_tensor_debug_info' in edge_attrs:
        input_data_node['fw_tensor_debug_info'] = edge_attrs['fw_tensor_debug_info']

    log.debug('Input: {} for node {}'.format(input_node.id, node_id))
    log.debug("Add edge from {} to {}".format(input_node.id, input_data_node.id))
    log.debug("Add edge from {} to {}".format(input_data_node.id, node_id))
    return input_node.id


def add_input_op_output_port_without_data(graph: Graph, node_id: str, input_op, port: int, fw_info: list):
    input_node = input_op.create_node()
    # In this case it can be more than one out edge from one port and we should iterate over all output edges
    for _, out_node, attrs in graph.out_edges(node_id, data=True):
        if attrs['out'] == port:
            # new out port = 0
            attrs = attrs.copy()
            attrs['out'] = 0
            attrs['fw_tensor_debug_info'] = fw_info
            attrs['data_attrs'] = ['fw_tensor_debug_info']
            graph.add_edge(input_node.id, out_node, **attrs)
            log.debug('Input: {} for node {} output port {}'.format(input_node.id, node_id, port))
            log.debug("Add edge from {} to {}".format(input_node.id, out_node))
    return input_node.id


def add_input_op_output_port_with_data(graph: Graph, node_id: str, input_op, port: int, fw_info: list):
    # we assume that after op always data node
    assert graph.stage == 'middle', 'add_input_op_input_port_with_data() function can be used only for graph after ' \
                                    'shape inference!'
    data_node = Node(graph, node_id).out_node(port)
    data_node['fw_tensor_debug_info'] = fw_info
    assert data_node.has_valid('kind') and data_node.kind == 'data'
    input_node = input_op.create_node()
    Node(graph, node_id).out_port(port).get_connection().set_source(input_node.out_port(0))
    log.debug('Input: {} for node {}'.format(input_node.id, node_id))
    log.debug("Add edge from {} to {}".format(input_node.id, node_id))
    return input_node.id


def add_input_op(graph: Graph, node_id: str, port: int = 0, data: bool = False,
                 shape=None, user_shape=None, data_type=None, is_out_port: bool = False):
    """
    This function adds Input node to node with id==node_id to specified port (in or out defined with is_out_port).
    :param graph: graph to operate on.
    :param node_id: node_id for node to which we should add new input.
    :param port: number of port of node_id node for adding input node.
    :param data: flag that define whether data nodes is needed or not.
    :param shape: shape for new input node.
    :param user_shape: shape provided by user which may contain boundaries of dynamic dimension.
    :param data_type: data type of input node.
    :param is_out_port: flag that define whether port is output port or not.
    :return: id of new Input operation
    """
    # We import it here because Op imports add_attrs_props and update_ie_fields from this file
    from openvino.tools.mo.ops.parameter import Parameter
    if data_type is None:
        data_type = np.float32
    input_op = Parameter(graph, dict(shape=shape, user_shape=user_shape, data_type=data_type, initial_node_name=node_id,
                                     name=get_new_placeholder_name(node_id, is_out_port, port)))

    if is_out_port:
        tensor_name = Node(graph, node_id).soft_get('name') + ":" + str(port)
    else:
        tensor_name = str(port) + ":" + Node(graph, node_id).soft_get('name')
    fw_info = [(Node(graph, node_id).soft_get('name'), tensor_name)]

    if not is_out_port and port == 0:
        tensor_name_no_port = Node(graph, node_id).soft_get('name')
        if graph.has_tensor_name(tensor_name_no_port):
            log.warning('Could not add user defined input name {} to tensor names list of as '
                        'graph contains tensor name with same name.'.format(tensor_name_no_port))
        else:
            # Add alias with operation name, as this format is used in some config files
            fw_info.append((Node(graph, node_id).soft_get('name'), tensor_name_no_port))

    edge_attrs = {'in': port, 'out': 0, 'in_attrs': ['in'], 'out_attrs': ['out'],
                  'fw_tensor_debug_info': fw_info,
                  'data_attrs': ['fw_tensor_debug_info']}

    if not data:
        if is_out_port:
            new_input_id = add_input_op_output_port_without_data(graph=graph, node_id=node_id, input_op=input_op,
                                                                 port=port, fw_info=edge_attrs['fw_tensor_debug_info'])
        else:
            new_input_id = add_input_op_input_port_without_data(graph=graph, node_id=node_id, input_op=input_op,
                                                                edge_attrs=edge_attrs)
    else:
        if is_out_port:
            new_input_id = add_input_op_output_port_with_data(graph=graph, node_id=node_id, input_op=input_op,
                                                              port=port, fw_info=edge_attrs['fw_tensor_debug_info'])
        else:
            new_input_id = add_input_op_input_port_with_data(graph=graph, node_id=node_id, input_op=input_op,
                                                             edge_attrs=edge_attrs)
    return new_input_id


def add_input_ops_helper_before_infer_input_port(graph: Graph, smart_node: Node, port: int, node_id: str,
                                                 shape: np.array, user_shape: tuple, data_type,
                                                 inputs: list, edges_to_remove: list):
    n_inputs = len(smart_node.in_nodes())
    if n_inputs > 1 and port is None:
        raise Error(
            'Node {} has more than 1 input and input shapes were provided. Try not to provide input'
            ' shapes or specify input port with port:node notation, where port is an integer. '
            '{}'.format(smart_node.soft_get('name'), refer_to_faq_msg(30)))
    port = port if port is not None else 0
    edges_to_remove.append((smart_node.in_node(port).id, smart_node.id))
    inputs.append(add_input_op(graph=graph, node_id=node_id, port=port, data=False,
                               shape=shape, user_shape=user_shape, data_type=data_type))


def add_input_ops_helper_after_infer_input_port(graph: Graph, smart_node: Node, port:int, node_id: str,
                                                inputs: list, edges_to_remove: list):
    port = port if port is not None else 0
    in_node = smart_node.in_node(port)
    shape = in_node['shape'] if 'shape' in in_node else None
    if shape is None:
        raise Error('Shape for tensor "{}" is not defined. Can not proceed.' + refer_to_faq_msg(41),
                    in_node.soft_get('name'))
    inputs.append(add_input_op(graph=graph, node_id=node_id, port=port, data=True,
                               shape=shape.copy(), data_type=in_node.soft_get('data_type', None)))
    edges_to_remove.append((in_node.id, node_id))


def add_input_ops_helper_before_infer_output_port(graph: Graph, port: int, node_id: str,
                                                  shape: np.array, user_shape: tuple, data_type: tuple,
                                                  inputs: list, edges_to_remove: list):
    for u, v, edge_attrs in graph.out_edges(node_id, data=True):
        if edge_attrs['out'] == port:
            edges_to_remove.append((u, v))  # we need to remove all edges from this port
    inputs.append(add_input_op(graph=graph, node_id=node_id, port=port, data=False,
                               shape=shape, user_shape=user_shape, data_type=data_type, is_out_port=True))


def add_input_ops_helper_after_infer_output_port(graph: Graph, smart_node: Node, port:int, node_id: str,
                                                 inputs: list, edges_to_remove: list):
    out_node = smart_node.out_node(port)
    shape = out_node['shape'] if 'shape' in out_node else None
    if shape is None:
        raise Error('Shape for tensor "{}" is not defined. Can not proceed.' + refer_to_faq_msg(41),
                    out_node.soft_get('name'))
    inputs.append(add_input_op(graph=graph, node_id=node_id, port=port, data=True,
                               shape=shape.copy(), data_type=out_node.soft_get('data_type', None), is_out_port=True))
    edges_to_remove.append((node_id, out_node.id))


def add_input_ops(graph: Graph, user_defined_inputs: dict, before_infer: bool):
    """
    This function add user defined input operations.
    For cutting without port:
    Op_1 -> Op_2 -> output, user_defined_inputs = {'Op_2': {'shape': PartialShape([1, 2])}} =>
    Op_1,  New_input (op=Parameter, shape=[1, 2]) -> Op_2 -> output

    For cutting with input port:
    Op_1 -> Op_2 -> output, user_defined_inputs = {'Op_2': {'shape':PartialShape([1, 2]), 'in': 0}} =>
    Op_1,  New_input (op=Parameter, shape=[1, 2]) -> Op_2 -> output

    For cutting with output port:
    Op_1 -> Op_2 -> output, user_defined_inputs = {'Op_2': {'shape':PartialShape([1, 2]), 'out': 0}} =>
    Op_1 -> Op_2, New_input (op=Parameter, shape=[1, 2]) -> output

    For case with before_infer=False data nodes are added to this schemes.
    """
    inputs = []
    set_is_input(graph, graph.get_nodes_with_attributes(op='Parameter'), False)
    if user_defined_inputs is None:
        inputs = graph.get_nodes_with_attributes(op='Parameter')
    else:
        # cutting the net by inputs
        assert isinstance(user_defined_inputs, dict)
        edges_to_remove = []
        for node_id in user_defined_inputs:
            for port_and_shape_info in user_defined_inputs[node_id]:
                if 'added' in port_and_shape_info and port_and_shape_info['added']:
                    continue

                is_out_port = 'out' in port_and_shape_info  # by default we assume input port or input node without port
                shape = port_and_shape_info['shape'] if 'shape' in port_and_shape_info else None
                user_shape = None
                if shape is not None:
                    user_shape = shape
                    shape_list = []
                    for dim in shape:
                        if isinstance(dim, Dimension):
                            if dim.is_static:
                                shape_list.append(dim.get_min_length())
                            else:
                                shape_list.append(dynamic_dimension_value)
                            continue
                        if dim >= 0:
                            shape_list.append(dim)
                        else:
                            shape_list.append(dynamic_dimension_value)

                    shape = shape_array(shape_list)
                data_type = port_and_shape_info['data_type'] if 'data_type' in port_and_shape_info else None
                smart_node = Node(graph, node_id)

                # Common port index check
                if is_out_port:
                    port = port_and_shape_info['out']  # we check that 'out' in port_and_shape_info earlier
                    if port is None:
                        raise Error('Output port for input node {} should be specified, it cannot be None!'.format(
                            node_id
                        ))
                    if port is not None and port not in smart_node.out_nodes():
                        raise Error('Output port index {} is out of number of available output ports for node "{}". ' +
                                    refer_to_faq_msg(29), port, node_id)
                else:
                    port = port_and_shape_info['in'] if 'in' in port_and_shape_info else None
                    if port is not None and port not in smart_node.in_nodes():
                        raise Error('Input port index {} is out of number of available input ports for node "{}". ' +
                                    refer_to_faq_msg(29), port, node_id)

                # specific Parameter case
                if smart_node.op == 'Parameter':
                    if port is not None:
                        raise Error(
                            'Parameter node "{}" doesn\'t have input port, but input port {} was provided. ' +
                            refer_to_faq_msg(28), node_id, port)
                    if shape is not None:
                        smart_node['shape'] = shape
                        smart_node['user_shape'] = user_shape
                    if data_type is not None:
                        smart_node['data_type'] = data_type
                    inputs.append(node_id)
                    port_and_shape_info['added'] = True

                    if smart_node.out_edges():
                        # User specified input is Parameter, so input cut is not needed, but
                        # Op name needs to be added to tensor names
                        op_name = smart_node.soft_get('name')
                        if graph.has_tensor_name(op_name):
                            continue
                        out_edges = list(graph.out_edges(op_name, data=True))
                        for _, _, attrs in out_edges:
                            fw_info = []
                            if 'fw_tensor_debug_info' in attrs:
                                fw_info += attrs['fw_tensor_debug_info']
                            attrs['fw_tensor_debug_info'] = fw_info + [(op_name, op_name)]

                    continue

                if before_infer:
                    if shape is None:
                        continue
                    # We cut with shapes provided by user and there is no need to wait till infer
                    if is_out_port:
                        add_input_ops_helper_before_infer_output_port(graph, port, node_id, shape, user_shape,
                                                                      data_type, inputs, edges_to_remove)
                    else:
                        add_input_ops_helper_before_infer_input_port(graph, smart_node, port, node_id, shape,
                                                                     user_shape, data_type, inputs,
                                                                     edges_to_remove)
                else:
                    # We cut after infer and we need inferred shapes in nodes
                    if is_out_port:
                        add_input_ops_helper_after_infer_output_port(graph, smart_node, port, node_id, inputs,
                                                                     edges_to_remove)
                    else:
                        add_input_ops_helper_after_infer_input_port(graph, smart_node, port, node_id, inputs,
                                                                    edges_to_remove)
                port_and_shape_info['added'] = True
        graph.remove_edges_from(edges_to_remove)

    # if len(inputs) == 0, shapes were not provided for all nodes in input-cut request,
    # we didn't cut inputs before infer, so this check is useless and invalid
    if len(inputs):
        set_is_input(graph, inputs, True)
        # Check if there are inputs that are not listed in user_defined_inputs and are needed to calculate outputs
        outputs = graph.get_nodes_with_attributes(op='Result')
        visited = set()
        for output_name in outputs:
            reverse_dfs(graph, output_name, check_input, visited)

    return inputs


class FrontExtractorOp(object):
    """
    A super class for an operation extractor.
    Do additional extraction of operation attributes without modifying of graph topology.
    Useful for custom layers that maps to a single FW operation to re-use of FW shape inference.
    In contrast to FrontReplacement* classes, this class doesn't modify graph topology and
    doesn't completely override node attributes. So it is safe to preserve the original
    MO inference function (which can use FW fallback mechanism).

    A sub-class should implement one of extract methods:
        def extract(self, node):
            return (<supported or not: Boolean>, { <additional node attributes> })
    """

    registered_ops = {}
    registered_cls = []

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.EXTRACTOR


class CaffePythonFrontExtractorOp:
    """
    A super class for custom caffe operation extractor.
    Do additional extraction of Python Caffe operation attributes without modifying the graph topology.
    Useful for Python layers that maps to a single FW operation to re-use of FW shape inference.
    In contrast to FrontReplacement* classes, this class doesn't modify graph topology and
    doesn't completely override node attributes. So it is safe to preserve the original
    MO inference function (which can use FW fallback mechanism).

    It is needed to keep the list of extractors for particularly Python layers.

    When actual extraction happens, Model Optimizer first finds the match by type, which is PythonFrontExtractorOp.
    It in turns looks up the CaffePythonFrontExtractorOp for the needed layer extractor not by type, but by
    the compilation of the layer name and the module name.

    A sub-class should implement one of extract methods:
        def extract(self, node):
            return (<supported or not: Boolean>, { <additional node attributes> })
    """
    registered_ops = {}
    registered_cls = []

    @staticmethod
    def get_attrs(pb) -> dict:
        params = pb.python_param
        attrs = CaffePythonFrontExtractorOp.parse_param_str(params.param_str)
        return attrs

    @staticmethod
    def parse_param_str(param_str: str) -> dict:
        if param_str[0] != '{' and param_str[-1] != '}':
            param_str = '{' + param_str + '}'
        return ast.literal_eval(param_str)

    @staticmethod
    def check_param(op_cls, attrs):
        for a in attrs:
            if a not in op_cls.supported_attrs(op_cls):
                log.error('Parameter {} is not recognised, please check correctness.\n List of supported parameters '
                          'is: {}'.format(a, op_cls.supported_attrs(op_cls)), extra={'is_warning': True})

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.EXTRACTOR
