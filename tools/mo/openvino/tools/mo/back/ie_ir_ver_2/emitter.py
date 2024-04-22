# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib

import defusedxml.ElementTree as ET
from defusedxml import defuse_stdlib
from defusedxml.minidom import parseString

from openvino.tools.mo.front.common.partial_infer.utils import unmask_shape, is_fully_defined
from openvino.tools.mo.graph.graph import *
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_precision
from openvino.tools.mo.utils.unsupported_ops import UnsupportedOps
from openvino.tools.mo.utils.utils import refer_to_faq_msg
from openvino.tools.mo.utils.version import get_version

# defuse_stdlib provide patched version of xml.etree.ElementTree which allows to use objects from xml.etree.ElementTree
# in a safe manner without including unsafe xml.etree.ElementTree
ET_defused = defuse_stdlib()[ET]
Element = ET_defused.Element
SubElement = ET_defused.SubElement
tostring = ET_defused.tostring

elements_to_skip_during_serializing = ['inputs_list']


def serialize_constants(graph: Graph, bin_file_name: str, data_type=np.float32):
    """
    Found all data constants that has output edges with 'bin' attribute.
    Serialize content for such constants to a binary file with name bin_file_name in
    raw format. Save offset and length of serialized area in the file as 'offset' and 'size'
    attributes of data node.

    Args:
        @graph: input graph with op and data nodes
        @bin_file_name: path to file to write blobs to
        @data_type: numpy data type to convert all blob elements to

    """
    bin_hashes = {}
    with open(bin_file_name, 'wb') as bin_file:
        serialize_constants_recursively(graph, bin_file, data_type, bin_hashes)


def update_offset_size_in_const_node(node: Node):
    assert node.kind == 'data'
    for consumer in node.out_nodes():
        if consumer.type != 'Const':
            continue
        assert not consumer.has_valid('offset')
        assert not consumer.has_valid('size')
        consumer['offset'] = node.offset
        consumer['size'] = node.size


def serialize_constants_recursively(graph: Graph, bin_file, data_type, bin_hashes):
    nodes = sorted(graph.nodes())
    for node in nodes:
        node = Node(graph, node)

        if node.kind == 'data' and node.value is not None and \
                any('bin' in d for u, v, d in graph.out_edges(node.node, data=True)):
            # avoid array copying while taking hash
            blob = node.value if node.value.ndim > 0 else node.value.reshape((1))
            assert is_fully_defined(blob), 'The constant value cannot contain dynamic values'
            if isinstance(blob, np.ma.masked_array):
                blob = np.ma.getdata(blob)
            blob_hash = hashlib.sha512(np.ascontiguousarray(blob).view(np.uint8)).hexdigest()

            if blob_hash in bin_hashes and np.array_equal(blob, bin_hashes[blob_hash]['blob']):
                graph.node[node.node]['offset'] = bin_hashes[blob_hash]['offset']
                graph.node[node.node]['size'] = bin_hashes[blob_hash]['size']
                graph.node[node.node]['blob_precision'] = np_data_type_to_precision(blob.dtype)
                update_offset_size_in_const_node(node)
            else:
                start = bin_file.tell()
                blob.tofile(bin_file)
                end = bin_file.tell()

                graph.node[node.node]['offset'] = start
                graph.node[node.node]['size'] = end - start
                graph.node[node.node]['blob_precision'] = np_data_type_to_precision(blob.dtype)

                bin_hashes[blob_hash] = {'offset': graph.node[node.node]['offset'],
                                         'size': graph.node[node.node]['size'], 'blob': blob}
                update_offset_size_in_const_node(node)

                assert (blob.dtype.itemsize * np.prod(node.shape) == end - start) or \
                       node.has_valid('force_shape'), node.attrs()

            log.debug(
                "Detected binary for graph: '{}', node: '{}', id: {}, shape: '{}', offset: '{}', size: '{}'".format(
                    graph, node.soft_get('name'), node.id, node.shape, node.offset, node.size))

    # separate loop for sub-graph to dump them after all blobs for more natural blob offset ordering
    # TODO: implement strict order for all blobs in entier IR
    for node in nodes:
        node = Node(graph, node)
        # Dump blobs recursively if sub-graphs are present in the node
        if node.has_valid('sub_graphs'):
            for sub_graph_attr_name in node.sub_graphs:
                sub_graph = node[sub_graph_attr_name]
                serialize_constants_recursively(sub_graph, bin_file, data_type, bin_hashes)


def serialize_mean_image(bin_file_name: str, mean_data=[]):
    with open(bin_file_name, 'ab') as bin_file:
        mean_offset = []
        mean_size = []
        for x in range(len(mean_data)):
            start = bin_file.tell()
            bin_file.write(mean_data[x][:])
            end = bin_file.tell()
            mean_offset.append(start)
            mean_size.append(end - start)

        return mean_offset, mean_size


def xml_shape(shape: np.ndarray, element: Element):
    for d in unmask_shape(shape):
        if d < -1:
            raise Error('The value "{}" for shape is not valid value.'.format(d))
        dim = SubElement(element, 'dim')
        if int(d) != d:
            raise Error('The value "{}" for shape is not integer.'.format(d))
        if not isinstance(d, np.int64):
            log.warning('The element of shape is not np.int64 value. Converting the value "{}" to integer'.format(d))
            d = int(d)
        dim.text = str(d)


def xml_ports(node: Node, element: Element, edges: Element):
    # input ports
    inputs = None  # will create input section only if at least one input is available
    for u, d in node.get_sorted_inputs():
        if 'bin' not in d and ('xml_skip' not in d or not d['xml_skip']):
            if inputs is None:
                inputs = SubElement(element, 'input')
            port = SubElement(inputs, 'port')
            port.set('id', str(d['in']))
            assert node.graph.node[u]['shape'] is not None, 'Input shape is not calculated properly for node {}'.format(
                node.id)
            xml_shape(node.graph.node[u]['shape'], port)

            # support saving rt_info passed from IR Reader
            port_id = d['in']
            if node.has('restored_input_ports') and port_id in node.restored_input_ports:
                port_rt_info_value = node.restored_input_ports[port_id][2]
                if port_rt_info_value != {}:
                    port_rt_info = SubElement(port, 'rt_info')
                    for (name, version), info_elem in port_rt_info_value.items():
                        attribute = SubElement(port_rt_info, 'attribute')
                        attribute.set('name', name)
                        attribute.set('version', str(version))
                        params = info_elem.serialize(node) if not isinstance(info_elem, dict) else info_elem
                        for key, value in params.items():
                            attribute.set(key, value)

            # u is a data node that has a single producer, let's find it
            assert (node.graph.node[u]['kind'] == 'data')
            in_nodes = list(node.graph.in_edges(u, data=True))
            assert (len(in_nodes) <= 1)
            if len(in_nodes) == 1:
                src, _, out_attrs = in_nodes[0]
                edge = SubElement(edges, 'edge')
                edge.set('from-layer', str(src))
                edge.set('from-port', str(out_attrs['out']))
                edge.set('to-layer', str(node.node))
                edge.set('to-port', str(d['in']))
                # port.set('precision', np_data_type_to_precision(node['_in_port_precision'][d['in']]))

    # output ports
    outputs = None
    for v, d in node.get_sorted_outputs():
        if 'xml_skip' not in d or not d['xml_skip']:
            if outputs is None:
                outputs = SubElement(element, 'output')
            port = SubElement(outputs, 'port')
            port.set('id', str(d['out']))
            # we need to check operation type, if it is const op, we don't renumber out ports
            # because they are already counted from zero
            port_id = d['out'] - len(node.in_nodes()) if node.type != 'Const' else d['out']
            data_type = node.out_port(port_id).get_data_type()
            assert data_type is not None, 'The precision is not defined for the output port {} of node {}' \
                                          ''.format(port_id, node.soft_get('name'))

            port.set('precision', node.soft_get('force_type', np_data_type_to_precision(data_type)))
            assert node.graph.node[v]['shape'] is not None, 'Output shape is not calculated properly for node {}' \
                                                            ''.format(node.id)
            tensor_names = node.out_port(port_id).get_tensor_names(port_renumber=True)
            if tensor_names:
                port.set('names', ','.join(tensor_names))
            xml_shape(node.graph.node[v]['shape'], port)

            # support saving rt_info passed from IR Reader
            if node.has('ports') and port_id in node.ports:
                port_rt_info_value = node.ports[port_id][2]
                if port_rt_info_value != []:
                    port_rt_info = SubElement(port, 'rt_info')
                    for (name, version), info_elem in port_rt_info_value.items():
                        attribute = SubElement(port_rt_info, 'attribute')
                        attribute.set('name', name)
                        attribute.set('version', str(version))
                        params = info_elem.serialize(node) if not isinstance(info_elem, dict) else info_elem
                        for key, value in params.items():
                            attribute.set(key, value)

def xml_consts(graph: Graph, node: Node, element: Element):
    blobs = None  # sub-element that will be created on-demand
    for u, d in node.get_sorted_inputs():
        if 'bin' in d and (node.type != 'Const'):
            if not blobs:
                blobs = SubElement(element, 'blobs')
            const = SubElement(blobs, d['bin'])
            try:
                const.set('offset', str(graph.node[u]['offset']))
                const.set('size', str(graph.node[u]['size']))
                const.set('precision', graph.node[u]['blob_precision'])
            except Exception as e:
                raise Error('Unable to access binary attributes ("offset" and/or "size") for blobs for node {}. '
                            'Details: {}'.format(node.soft_get('name'), e))


def soft_get(node, attr):
    """ If node has soft_get callable member, returns node.soft_get(attr), else return <SUB-ELEMENT> """
    return node.soft_get(attr) if hasattr(node, 'soft_get') and callable(node.soft_get) else '<SUB-ELEMENT>'


def serialize_element(
        graph: Graph,
        node,
        schema: list,
        parent_element: Element,
        edges: Element,
        unsupported):
    name, attrs, subelements = schema
    element = SubElement(parent_element, name)
    for attr in attrs:
        if isinstance(attr, tuple):
            key = attr[0]
            try:
                if callable(attr[1]):
                    value = attr[1](node)
                else:
                    value = node[attr[1]] if attr[1] in node else None
            except TypeError as e:
                raise Error('Unable to extract {} from layer {}', key, soft_get(node, 'name')) from e
            except Exception as e:
                raise Error(
                    'Cannot emit value for attribute {} for layer {}. '
                    'Internal attribute template: {}.',
                    key,
                    soft_get(node, 'name'),
                    attr
                ) from e
        elif isinstance(attr, dict):
            node_attrs = node.graph.node[node.id] if isinstance(node, Node) else node
            for key in attr.keys():
                if key in node_attrs:
                    for k, v in node_attrs[key].items():
                        element.set(k, str(v))
            continue
        else:
            key = attr
            value = node[attr] if attr in node else None
        if value is not None:
            element.set(key, str(value))
    serialize_node_attributes(graph, node, subelements, element, edges, unsupported)
    if len(element.attrib) == 0 and len(list(element)) == 0:
        parent_element.remove(element)


def serialize_meta_list(graph, node, schema, element, edges, unsupported):
    _, list_accessor, sub_schema = schema
    items = list_accessor(node)  # this is a list of dictionary-like objects
    for item in items:
        serialize_node_attributes(graph, item, [sub_schema], element, edges, unsupported)


def serialize_runtime_info(node, parent_element: Element):
    if 'rt_info' not in node:
        return
    rt_info = SubElement(parent_element, 'rt_info')

    for (name, version), info_elem in node.rt_info.info.items():
        attribute = SubElement(rt_info, 'attribute')
        attribute.set('name', name)
        attribute.set('version', str(version))
        params = info_elem.serialize(node) if not isinstance(info_elem, dict) else info_elem
        for key, value in params.items():
            attribute.set(key, value)
    if len(rt_info.attrib) == 0 and len(list(rt_info)) == 0:
        parent_element.remove(rt_info)


def serialize_node_attributes(
        graph: Graph,  # the current network graph
        node,  # dictionary-like object that should be serialized
        schema: list,
        parent_element: Element,
        edges: Element,
        unsupported):
    # the Result op may be marked so it should not appear in the IR. For example, refer to transformation
    # openvino/tools/mo/back/TopKNormalizer.py
    if isinstance(node, Node) and node.soft_get('type') == 'Result' and node.has_and_set('keep_output_port'):
        return
    try:
        for s in schema:
            if not isinstance(s, tuple):
                if s == '@ports':
                    try:
                        # TODO make sure that edges are generated regardless of the existence of @ports
                        xml_ports(node, parent_element, edges)
                    except Exception as e:
                        raise Error(('Unable to create ports for node with id {}. ' +
                                     refer_to_faq_msg(3)).format(node.id)) from e
                elif s == '@consts':
                    xml_consts(graph, node, parent_element)
                elif s == '@runtime_info':
                    serialize_runtime_info(node, parent_element)
                else:
                    log.warning('Unknown xml schema tag: {}'.format(s))
            else:
                name = s[0]
                if name == '@list':
                    serialize_meta_list(graph, node, s, parent_element, edges, unsupported)
                elif name == '@network':
                    serialize_network(node[s[1]], parent_element, unsupported)
                else:
                    serialize_element(graph, node, s, parent_element, edges, unsupported)
    except Exception as e:
        raise Error(
            'Error while emitting attributes for layer {} (id = {}). It usually means that there is unsupported '
            'pattern around this node or unsupported combination of attributes.',
            soft_get(node, 'name'),
            node.id
        ) from e


def create_pre_process_block_for_image(net: Element, ref_layer_names: list, mean_offset: tuple,
                                       mean_size: tuple):
    pre_process = SubElement(net, 'pre-process')
    pre_process.set('mean-precision', 'FP32')  # TODO: to think about need to output FP16 mean values
    # TODO: extend it for several inputs
    pre_process.set('reference-layer-name', ref_layer_names[0])
    for idx in range(len(mean_size)):
        channel_xml = SubElement(pre_process, 'channel')
        channel_xml.set('id', str(idx))
        mean_xml = SubElement(channel_xml, 'mean')
        mean_xml.set('offset', str(mean_offset[idx]))
        mean_xml.set('size', str(mean_size[idx]))


def create_pre_process_block(net, ref_layer_name, means, scales=None):
    """
    Generates the pre-process block for the IR XML
    Args:
        net: root XML element
        ref_layer_name: name of the layer where it is referenced to
        means: tuple of values
        scales: tuple of values

    Returns:
        pre-process XML element
    """
    pre_process = SubElement(net, 'pre-process')
    pre_process.set('reference-layer-name', ref_layer_name)

    for idx in range(len(means)):
        channel_xml = SubElement(pre_process, 'channel')
        channel_xml.set('id', str(idx))

        mean_xml = SubElement(channel_xml, 'mean')
        mean_xml.set('value', str(means[idx]))

        if scales:
            scale_xml = SubElement(channel_xml, 'scale')
            scale_xml.set('value', str(scales[idx]))

    return pre_process


def add_quantization_statistics(graph, net_element):
    if 'statistics' in graph.graph:
        stats = SubElement(net_element, 'statistics')
        for tensor, interval in graph.graph['statistics'].items():
            layer = SubElement(stats, 'layer')
            name = SubElement(layer, 'name')
            name.text = tensor
            min = SubElement(layer, 'min')
            min.text = interval['min']
            max = SubElement(layer, 'max')
            max.text = interval['max']
        log.info('Statistics were inserted to IR')


def add_quantization_info_section(net: Element, meta_info: dict):
    if 'quantization_parameters' in meta_info:
        parameters = meta_info['quantization_parameters']
        quant_params = SubElement(net, 'quantization_parameters')

        config = SubElement(quant_params, 'config')
        config.text = parameters['config']

        version = SubElement(quant_params, 'version')
        version.set('value', parameters['version'])

        cli_params = SubElement(quant_params, 'cli_params')
        cli_params.set('value', parameters['cli_params'])


def add_meta_data_elem(meta: Element, key, value):
    if isinstance(value, dict):
        sub_elem = SubElement(meta, key)
        for sub_key, sub_value in sorted(value.items()):
            if sub_value in elements_to_skip_during_serializing:
                continue
            add_meta_data_elem(sub_elem, sub_key, sub_value)
    else:
        SubElement(meta, key).set('value', str(value))


def add_net_rt_info(net: Element, meta_info: dict):
    if meta_info == {}:
        log.warning('`meta_info` is not provided, IR will not contain appropriate section.')
    else:
        meta = SubElement(net, 'rt_info')
        for key, value in meta_info.items():
            if isinstance(value, dict) and value == {}:
                continue
            add_meta_data_elem(meta, key, value)


def serialize_node(graph: Graph, node: Node, layers: SubElement, edges: SubElement, unsupported: UnsupportedOps):
    if node.kind == 'op' and (not node.has('type') or node.type is None):
        unsupported.add(node)
        return
    if not node.has('IE'):
        return
    try:
        serialize_node_attributes(graph, node, node.IE, layers, edges, unsupported)
    except Error as e:
        raise Error(str(e).replace('<SUB-ELEMENT>', '{} (id = {})'.format(node.soft_get('name'), node.id))) from e


def get_tensor_names_of_result_node(graph):
    result_nodes = graph.get_op_nodes(type='Result')
    result_names_to_tensor_names = {}
    for res_node in result_nodes:

        # After port renumbering port/connection API is not applicable
        assert len(res_node.in_nodes()) > 0, \
            "Result node with name {} has no input node.".format(res_node.soft_get('name'))
        res_data_node = res_node.in_node(0)
        assert len(res_data_node.in_nodes()) > 0, \
            "Data node of Result with name {} has no input node.".format(res_node.soft_get('name'))
        res_in_node = res_data_node.in_node(0)

        # We cannot use out_ports() after port renumbering
        for v, d in res_in_node.get_sorted_outputs():
            port_id = d['out'] - len(res_in_node.in_nodes()) if res_in_node.type != 'Const' else d['out']
            tensor_names = res_in_node.out_port(port_id).get_tensor_names(port_renumber=True)
            result_names_to_tensor_names[res_node.soft_get('name')] = tensor_names
    return result_names_to_tensor_names


def find_result_node_by_name(output_name, result_nodes, result_names_to_tensor_names):
    for res_node in result_nodes:
        res_name = res_node.soft_get('name')
        tensor_names = result_names_to_tensor_names[res_name]
        if output_name in tensor_names:
            # In this case output tensor name is in tensor names list of previous op
            return res_name

    return None

def check_and_add_result_name(result_name:str, ordered_results:list):
    if result_name in ordered_results:
        log.warning("Result node with name {} has at least two tensor names corresponding "
                    "to different original results.".format(result_name))
    else:
        ordered_results.append(result_name)

def serialize_network(graph, net_element, unsupported):
    layers = SubElement(net_element, 'layers')
    edges = SubElement(net_element, 'edges')
    if graph is None:
        return
    nodes = sorted(graph.nodes())

    result_nodes = graph.get_op_nodes(type='Result')
    result_names_to_tensor_names = get_tensor_names_of_result_node(graph)

    ordered_results = []
    for output_name in graph.outputs_order:
        node = graph.get_op_nodes(name=output_name)

        if len(node) == 0:
            # As graph does not contain node with name=output_name
            # in the following code we look for output_name among tensor names
            # incoming to Result nodes
            found_result_name = find_result_node_by_name(output_name, result_nodes, result_names_to_tensor_names)

            if found_result_name is not None:
                check_and_add_result_name(found_result_name, ordered_results)
            else:
                log.warning("Output node with name {} is not found in graph.".format(output_name))
            continue
        node = node[0]

        # In this case Result node has the same name as output tensor
        if node.soft_get('type') == 'Result':
            check_and_add_result_name(node.soft_get('name'), ordered_results)
            continue

        # Here output data node count is checked. Output Op nodes must have at least one data node
        assert len(node.out_nodes()) >= 1, "Incorrect graph. Non-Result node with name {} " \
                                           "has no output data node.".format(output_name)

        # After port renumbering port/connection API is not applicable, and output port numbering
        # starts from len(node.in_nodes()). But it not applicable to Constant operations, they have only one output
        # port with number 0.
        if node.type == 'Const':
            data_node = node.out_node(0)
        else:
            data_node = node.out_node(len(node.in_nodes()))

        found_result = False
        for op_node in data_node.out_nodes():
            if op_node.soft_get('type') == 'Result':
                found_result = True
                check_and_add_result_name(op_node.soft_get('name'), ordered_results)
                break

        if not found_result:
            log.warning("Node that expected to be output with name {} is not connected with Result node.".format(output_name))

    param_nodes = graph.get_op_nodes(type='Parameter')
    serialized_inputs = []
    for input_name in graph.inputs_order:
        node = graph.get_op_nodes(name=input_name)
        if len(node) != 0:
            serialize_node(graph, node[0], layers, edges, unsupported)
            serialized_inputs.append(input_name)
            continue
        found_tensor_name = False
        for param_node in param_nodes:
            param_name = param_node.soft_get('name')
            if not param_node.is_out_port_connected(0):
                continue
            tensor_names = param_node.out_port(0).get_tensor_names(port_renumber=True)
            if input_name in tensor_names:
                # In this case input name is in tensor names list of Parameter op
                serialize_node(graph, param_node, layers, edges, unsupported)
                serialized_inputs.append(param_name)
                found_tensor_name = True
                break

        if not found_tensor_name:
            log.warning("Input node with name {} is not found in graph.".format(param_name))

    for node in nodes:
        node = Node(graph, node)
        if node.soft_get('name') in serialized_inputs:
            continue
        if node.soft_get('name') in ordered_results:
            continue
        serialize_node(graph, node, layers, edges, unsupported)

    for output_name in ordered_results:
        node = graph.get_op_nodes(name=output_name)
        assert len(node) == 1, "Output node with name {} is not found in graph.".format(output_name)
        serialize_node(graph, node[0], layers, edges, unsupported)


def generate_ie_ir(graph: Graph, file_name: str, input_names: tuple = (), mean_offset: tuple = (),
                   mean_size: tuple = (), meta_info: dict = dict()):
    """
    Extracts OV/IR attributes from kind='op' nodes in three ways:
      (1) node.OV xml scheme that sets correspondence from existing attributes to generated xml elements
      (2) input/output edges that don't have 'bin' attributes are transformed to input/output ports
      (3) input edges that has 'bin' attributes are handled in special way like weights/biases

    Args:
        graph: nx graph with FW-independent model
        file_name: name of the resulting IR
        input_names: names of input layers of the topology to add mean file to
        input_name: name of the layer which is referenced from pre-processing block if any
        mean_values: tuple of mean values for channels in RGB order
        scale_values:  tuple of mean values for channels in RGB order
        mean_offset: offset in binary file, where mean file values start
        mean_size: size of the mean file
    """
    net = Element('net')
    net.set('name', graph.name)
    net.set('version', str((graph.graph['ir_version'])))

    if mean_size or mean_offset:
        create_pre_process_block_for_image(net, input_names, mean_offset, mean_size)

    if 'mean_values' in graph.graph.keys():
        for input_name, values in graph.graph['mean_values'].items():
            create_pre_process_block(net, input_name, values)

    unsupported = UnsupportedOps(graph)

    serialize_network(graph, net, unsupported)

    #TODO: Remove this line when POT updates to using of rt_info
    add_quantization_statistics(graph, net)

    add_net_rt_info(net, meta_info)

    #TODO: Remove this line when POT updates to using of rt_info
    add_quantization_info_section(net, meta_info)

    xml_string = tostring(net)
    xml_doc = parseString(xml_string)
    pretty_xml_as_string = xml_doc.toprettyxml()
    if len(unsupported.unsupported):
        log.debug('Partially correct IR XML:\n{}'.format(pretty_xml_as_string))
        unsupported.report(log.error, "List of operations that cannot be converted to OpenVINO IR:")
        raise Error('Part of the nodes was not converted to IR. Stopped. ' +
                    refer_to_faq_msg(24))
    with open(file_name, 'wb') as file:
        file.write(bytes(pretty_xml_as_string, "UTF-8"))


def port_renumber(graph: Graph):
    for node in graph.get_op_nodes():
        base = 0
        # we need to check operation type, if it is const op, we don't renumber out ports to count them from zero
        if node.soft_get('type') != 'Const':
            for u, d in node.get_sorted_inputs():
                d['in'] = base
                base += 1
        for v, d in node.get_sorted_outputs():
            d['out'] = base
            base += 1
