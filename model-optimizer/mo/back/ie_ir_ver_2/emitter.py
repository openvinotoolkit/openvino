"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import xml.dom.minidom
from xml.etree.ElementTree import Element, SubElement, tostring

from mo.front.extractor import update_ie_fields
from mo.graph.graph import *
from mo.utils.unsupported_ops import UnsupportedOps
from mo.utils.utils import refer_to_faq_msg


def create_const_nodes(graph: nx.MultiDiGraph):
    """
    Adds layers with type 'Const' that produce blob from 'bin' file. The pass finds data nodes with one output which
    doesn't have edge with 'bin' attribute and generate Const op node before the node and data node before the Const
    node. The data node before 'Const' node is needed because the op node dumps input tensors to bin file.
    :param graph: input graph.
    :return: None
    """
    for node_name in list(graph.nodes()):
        node = NodeWrap(graph, node_name)
        if (
                node.has('kind') and
                node.kind == 'data' and (
                (len(node.out_edges()) == 1 and 'bin' not in node.out_edge(0)) or
                node.has_and_set('is_output')
        ) and
                len(node.in_nodes()) == 0):

            if node.has_valid('value'):
                const_node_name = node.id + '_const'
                log.debug("Added Const node '{}'".format(const_node_name))
                graph.add_node(const_node_name, name=const_node_name, type='Const', kind='op', op='Const',
                               precision="FP32")
                update_ie_fields(node.graph.node[const_node_name])
                graph.add_edges_from([(const_node_name, node.id, {'out': 0})])
                copy_data_node_name = unique_id(graph, node.id + '_copy_')
                graph.add_node(copy_data_node_name, kind='data', precision="FP32", shape=np.array(node.shape),
                               value=np.array(node.value))
                if node.has_valid('force_precision'):
                    Node(graph, copy_data_node_name)['force_precision'] = node.force_precision
                    Node(graph, const_node_name)['force_precision'] = node.force_precision
                graph.add_edges_from([(copy_data_node_name, const_node_name, {'in': 0, 'bin': 'custom'})])
            else:
                log.debug('node = {}'.format(node.graph.node[node.id]))
                raise Error(
                    'Discovered data node without inputs and value, node.name = {}, consumer.name = {}. ' +
                    refer_to_faq_msg(23),
                    node.soft_get('name'),
                    node.out_node().soft_get('name') if len(node.out_nodes()) else "<no consumer>"
                )


def serialize_constants(graph: nx.MultiDiGraph, bin_file_name: str, data_type=np.float32):
    """
    Found all data constants that has output edges with 'bin' attribute.
    Serialize content for such constants to a binary file with name bin_file_name in
    raw format. Save offset and length of serialized area in the file as 'offset' and 'size'
    attributes of data node.

    Args:
        @graph: input graph with op and data nodes
        @bin_file_name: path to file to write blobs to
        @data_type: numpy data type to convert all blob elemnts to

    """
    with open(bin_file_name, 'wb') as bin_file:
        nodes = sorted(graph.nodes())
        for node in nodes:
            node = NodeWrap(graph, node)
            if node.kind != 'data' or node.value is None:
                continue

            if any('bin' in d for u, v, d in graph.out_edges(node.node, data=True)):
                blob = node.value
                start = bin_file.tell()
                blob.tofile(bin_file)
                end = bin_file.tell()

                graph.node[node.node]['offset'] = start
                graph.node[node.node]['size'] = end - start
                assert (blob.dtype.itemsize * np.prod(node.shape) == end - start)
                log.debug("Detected binary for shape: '{}' offset: '{}' size: '{}'".format(node.shape, node.offset,
                                                                                           node.size))


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


def xml_shape(shape: np.ndarray, element: xml.etree.ElementTree.Element):
    for d in shape:
        dim = SubElement(element, 'dim')
        dim.text = str(d)


def xml_ports(node: Node, element: xml.etree.ElementTree.Element, edges: xml.etree.ElementTree.Element):
    # input ports
    inputs = None  # will create input section only if at least one input is available
    for u, d in get_sorted_inputs(node):
        if 'bin' not in d and ('xml_skip' not in d or not d['xml_skip']):
            if inputs is None:
                inputs = SubElement(element, 'input')
            port = SubElement(inputs, 'port')
            port.set('id', str(d['in']))
            assert node.graph.node[u]['shape'] is not None, 'Input shape is not calculated properly for node {}'.format(
                node.id)
            xml_shape(node.graph.node[u]['shape'], port)
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

    # output ports
    outputs = None
    for v, d in get_sorted_outputs(node):
        if 'xml_skip' not in d or not d['xml_skip']:
            if outputs is None:
                outputs = SubElement(element, 'output')
            port = SubElement(outputs, 'port')
            port.set('id', str(d['out']))
            assert node.graph.node[v][
                       'shape'] is not None, 'Output shape is not calculated properly for node {}'.format(
                node.id)
            xml_shape(node.graph.node[v]['shape'], port)


def xml_consts(node: Node, element: xml.etree.ElementTree.Element):
    blobs = None  # sub-element that will be created on-demand
    for u, d in get_sorted_inputs(node):
        if 'bin' in d:
            if not blobs:
                blobs = SubElement(element, 'blobs')
            const = SubElement(blobs, d['bin'])
            const.set('offset', str(node.graph.node[u]['offset']))
            const.set('size', str(node.graph.node[u]['size']))


def serialize_node_attributes(node: Node, schema: list, layers: xml.etree.ElementTree.Element,
                              edges: xml.etree.ElementTree.Element):
    try:
        for s in schema:
            if not isinstance(s, tuple):
                if s == '@ports':
                    try:
                        # TODO make sure that edges are generated regardless of the existence of @ports
                        xml_ports(node, layers,
                                  edges)
                    except Exception as e:
                        raise Error(('Unable to create ports for node with id {}. ' +
                                    refer_to_faq_msg(3)).format(node.id)) from e
                elif s == '@consts':
                    xml_consts(node, layers)
                else:
                    log.warning('Unknown xml schema tag: {}'.format(s))
            else:
                name, attrs, subelements = s
                element = SubElement(layers, name)
                for attr in attrs:
                    if isinstance(attr, tuple):
                        key = attr[0]
                        try:
                            if callable(attr[1]):
                                value = attr[1](node)
                            else:
                                value = node[attr[1]] if node.has(attr[1]) else None
                        except TypeError as e:
                            raise TypeError('Unable to extract {} from layer {}', key, node.soft_get('name')) from e
                        except Exception as e:
                            raise Error(
                                'Cannot emit value for attribute {} for layer {}. '
                                'Internal attribute template: {}.',
                                key,
                                node.soft_get('name'),
                                attr
                            ) from e
                    elif isinstance(attr, dict):
                        node_attrs = node.graph.node[node.id]
                        for key in attr.keys():
                            if key in node_attrs:
                                for k, v in node_attrs[key].items():
                                    element.set(k, str(v))
                        continue
                    else:
                        key = attr
                        value = node[attr] if node.has(attr) else None
                    if value is not None:
                        element.set(key, str(value))
                serialize_node_attributes(node, subelements, element, edges)
                if len(element.attrib) == 0 and len(element.getchildren()) == 0:
                    layers.remove(element)
    except Exception as e:
        raise Error(
            'Error while emitting attributes for layer {}. '
            'It usually means that there is unsupported pattern around this node or unsupported combination of attributes.',
            node.soft_get('name')
        ) from e


def create_pre_process_block_for_image(net: xml.etree.ElementTree.Element, ref_layer_names: list, mean_offset: tuple,
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


def generate_ie_ir(graph: nx.MultiDiGraph, file_name: str, input_names: tuple = (), mean_offset: tuple = (),
                   mean_size: tuple = ()):
    """
    Extracts IE/IR attributes from kind='op' nodes in three ways:
      (1) node.IE xml scheme that set correspondance from existing attributes to generated xml elements
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
    nodes = sorted(graph.nodes())
    net = Element('net')
    net.set('name', graph.name)
    net.set('version', '2')
    net.set('batch', '1')  # TODO substitute real batches here (is it a number or is it an index?)

    layers = SubElement(net, 'layers')
    edges = SubElement(net, 'edges')

    if mean_size or mean_offset:
        create_pre_process_block_for_image(net, input_names, mean_offset, mean_size)

    if 'mean_values' in graph.graph.keys():
        for input_name, values in graph.graph['mean_values'].items():
            create_pre_process_block(net, input_name, values)

    unsupported = UnsupportedOps(graph)

    for node in nodes:
        node = NodeWrap(graph, node)
        if not node.has('IE'):
            continue
        if node.kind == 'op' and (not node.has('type') or node.type is None):
            unsupported.add(node)
            continue
        serialize_node_attributes(node, node.IE, layers, edges)
    xml_string = tostring(net)
    xml_doc = xml.dom.minidom.parseString(xml_string)  # ugly?
    pretty_xml_as_string = xml_doc.toprettyxml()
    if len(unsupported.unsupported):
        log.debug('Partially correct IR XML:\n{}'.format(pretty_xml_as_string))
        unsupported.report(log.error, "List of operations that cannot be converted to IE IR:")
        raise Error('Part of the nodes was not translated to IE. Stopped. ' +
                    refer_to_faq_msg(24))
    with open(file_name, 'w') as file:
        file.write(pretty_xml_as_string)


def port_renumber(graph: nx.MultiDiGraph):
    for node in list(graph.nodes()):
        node = NodeWrap(graph, node)
        if node.kind == 'op':
            base = 0
            for u, d in get_sorted_inputs(node):
                d['in'] = base
                base += 1
            for v, d in get_sorted_outputs(node):
                d['out'] = base
                base += 1
