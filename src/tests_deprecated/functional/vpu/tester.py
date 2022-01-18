# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import errno
import os
import sys
from pathlib import Path
from argparse import ArgumentParser

from typing import Union

import xml.etree.cElementTree as et
import xml.dom.minidom as dom


def get_path(entry: Union[str, Path], is_directory=False, check_exists=True, file_or_directory=False):
    try:
        path = Path(entry)
    except TypeError:
        raise TypeError('"{}" is expected to be a path-like'.format(entry))

    if not check_exists:
        return path

    # pathlib.Path.exists throws an exception in case of broken symlink
    if not os.path.exists(str(path)):
        raise FileNotFoundError('{}: {}'.format(os.strerror(errno.ENOENT), path))

    if not file_or_directory:
        if is_directory and not path.is_dir():
            raise NotADirectoryError('{}: {}'.format(os.strerror(errno.ENOTDIR), path))

        # if it exists it is either file (or valid symlink to file) or directory (or valid symlink to directory)
        if not is_directory and not path.is_file():
            raise IsADirectoryError('{}: {}'.format(os.strerror(errno.EISDIR), path))

    return path


def build_argument_parser():
    parser = ArgumentParser(
        description='Tool for adding tests in OpenVINO project. It is intended to crop IR by given layer name for per-layer tests.',
        allow_abbrev=False
    )
    parser.add_argument(
        '-w', '--weights',
        help='path to the bin file containing weights (OpenVINO IR); '
             'if not specified, the value specified in "--model" option with "bin" extension is used',
        type=get_path,
        required=False
    )
    parser.add_argument(
        '-o', '--output',
        help='name of output files (.xml and .bin); '
             'if not specified, the value specified in "--model" option with "_extracted" suffix is used',
        type=str,
        required=False
    )

    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-m', '--model',
        help='path to the xml file containing model (OpenVINO IR); only IRv10 or newer is supported at the moment',
        type=get_path,
        required=True
    )
    required.add_argument(
        '-n', '--name',
        help='name of the layer to be tested',
        type=str,
        required=True
    )

    return parser


def tostring(layer, num_tabs=0):
    return '\t' * num_tabs + str(et.tostring(layer, encoding='utf-8').decode())


def tags(node):
    inner_tags = {}
    for tag in node:
        inner_tags[tag.tag] = tag
    return inner_tags


def make_input(layer, port_id):
    output = et.Element('output')
    output_port = next(port for port in tags(layer)['output'] if int(port.attrib['id']) == int(port_id))
    output.append(output_port)

    precision_to_element_type = {'FP16': 'f16', 'I32': 'i32', 'FP32': 'f32'}
    data = et.Element('data', attrib={
        'element_type': precision_to_element_type[output_port.attrib['precision']],
        'shape': ','.join([dim.text for dim in output_port])}
    )

    input_layer = et.Element('layer', attrib={
        'id': layer.attrib['id'],
        'name': layer.attrib['name'],
        'type': 'Parameter',
        'version': 'opset1'
    })
    input_layer.append(data)
    input_layer.append(output)

    return input_layer


def make_output(layer, port_id):
    input_section = et.Element('input')
    input_port = next(port for port in tags(layer)['input'] if int(port.attrib['id']) == int(port_id))
    input_section.append(input_port)

    output_layer = et.Element('layer', attrib={
        'id': layer.attrib['id'],
        'name': layer.attrib['name'],
        'type': 'Result',
        'version': 'opset1'
    })
    output_layer.append(input_section)

    return output_layer


def extract_weights(source, layers, destination):
    def update(section):
        offset = int(section.attrib['offset'])
        size = int(section.attrib['size'])

        src.seek(offset)
        original_segment = (offset, offset + size - 1)
        if original_segment in output_offsets:
            section.attrib['offset'] = str(output_offsets[original_segment])
        else:
            output_offset = dst.tell()
            dst.write(src.read(size))

            output_offsets[original_segment] = output_offset
            section.attrib['offset'] = str(output_offset)

    output_offsets = {}
    with Path(source).open(mode='rb') as src, Path(destination).open(mode='w+b') as dst:
        for layer in layers:
            for inner_layer in layer.iter('layer'):
                if inner_layer.attrib['version'] == 'opset1' and inner_layer.attrib['type'] == 'Const':
                    # for standard operation set IR v10 keeps weights as Const layers
                    # metadata such as offset and size is stored in data section
                    update(inner_layer.find('data'))
                else:
                    # for other operation sets previous way is used
                    # weights metadata is stored in blobs section
                    # with "custom", "weights" and "biases" subsections
                    blobs = inner_layer.find('blobs') or []
                    for blob in blobs:
                        update(blob)


def prettify(element, indent=0):
    header = dom.Document().toxml()
    string = dom.parseString(tostring(element)).toprettyxml()[len(header) + 1:]
    return '\n'.join(['\t' * indent + line for line in string.split('\n') if line.strip()])


def dump(input_model, elements, output_model):
    root = et.parse(str(input_model)).getroot()
    net = et.Element('net', attrib={'name': root.attrib['name'], 'version': root.attrib['version']})

    layers = et.Element('layers')
    for layer in elements['layers']:
        layers.append(layer)

    edges = et.Element('edges')
    for edge in elements['edges']:
        edges.append(edge)

    net.append(layers)
    net.append(edges)

    with Path(output_model).open(mode='w+t') as output:
        print('<?xml version="{}" ?>'.format(input_model.version), file=output)
        print(prettify(net), file=output)


def main():
    arguments = build_argument_parser().parse_args()
    model = dom.parse(str(input_model))
    if int(model.version) < 10:
        print('Error: only IR version 10 or newer is supported, IRv{} has been given'.format(model.version))
        sys.exit(-1)

    layers, edges, _ = et.parse(str(arguments.model)).getroot()
    layers_identifiers = {}
    for layer in layers:
        layers_identifiers[int(layer.attrib['id'])] = layer

    input_edges, output_edges = {}, {}
    for edge in edges:
        from_layer = int(edge.attrib['from-layer'])
        to_layer = int(edge.attrib['to-layer'])

        if to_layer not in input_edges:
            input_edges[to_layer] = [edge]
        else:
            input_edges[to_layer].append(edge)

        if from_layer not in output_edges:
            output_edges[from_layer] = [edge]
        else:
            output_edges[from_layer].append(edge)

    elements = {'layers': [], 'edges': []}
    layer = next(operation for operation in layers if operation.attrib['name'] == arguments.name)

    identifier = int(layer.attrib['id'])
    elements['edges'] = input_edges[identifier] + output_edges[identifier]

    for edge in input_edges[identifier]:
        input_layer = layers_identifiers[int(edge.attrib['from-layer'])]

        if input_layer.attrib['type'] != 'Const':
            elements['layers'].append(make_input(input_layer, int(edge.attrib['from-port'])))
        else:
            elements['layers'].append(input_layer)

    elements['layers'].append(layer)

    for edge in output_edges[identifier]:
        output_layer = layers_identifiers[int(edge.attrib['to-layer'])]
        if output_layer.attrib['type'] != 'Result':
            elements['layers'].append(make_output(output_layer, int(edge.attrib['to-port'])))
        else:
            elements['layers'].append(output_layer)

    weights = arguments.weights or str(arguments.model)[:-3] + 'bin'
    output = arguments.output or str(arguments.model)[:-4] + '_extracted'
    output_weights = Path(output + '.bin')
    extract_weights(weights, elements['layers'], output_weights)

    output_model = Path(output + '.xml')
    dump(model, elements, output_model)


if __name__ == '__main__':
    main()
