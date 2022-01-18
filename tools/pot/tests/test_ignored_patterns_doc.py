# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from openvino.tools.pot.configs import hardware
from openvino.tools.pot.graph.special_patterns import get_ignored_patterns


def dump_pattern_to_mermaid_flowchart(pattern):
    def _prepare_pattern(pattern):
        nodes = []
        for (node_id, node_data) in pattern['nodes']:
            if node_data['kind'] != 'data': # skip data nodes
                nodes.append((node_id, node_data))
        edges = []
        # remove edges with data nodes
        for (begin_1, end_1) in pattern['edges']:
            if 'data' in end_1:
                for (begin_2, end_2) in pattern['edges']:
                    if end_1 == begin_2:
                        edges.append((begin_1, end_2))
        return nodes, edges

    def _op_type_to_list(op_type):
        if isinstance(op_type, str):
            return [op_type]
        all_op_types = ['ReLU', 'PReLU', 'Activation', 'Sigmoid',
                        'Convolution', 'MatMul',
                        'ConvolutionBackpropData', 'GroupConvolution', 'GroupConvolutionBackpropData',
                        'Add', 'Multiply', 'ReduceMean', 'Squeeze']
        return filter(op_type, all_op_types)

    nodes, edges = _prepare_pattern(pattern)

    BR_TAG = '<br/>'
    SHIFT = '    ' # 4 spaces
    BEGIN_MERMAID_SECTION = '```mermaid'
    END_MERMAID_SECTION = '```'
    BEGIN_GRAPH = 'graph TB'
    DELIMITER = '---'
    QUANTIZED_NODE_COLOR = '#73C2FB'
    node_id_to_node_data = dict(nodes)

    pattern_attributes = [('Name', pattern['name']), ('Pattern', '')]

    markdown_lines = []
    for (attr_name, attr_value) in pattern_attributes:
        markdown_lines.append(f'**{attr_name}:** {attr_value}{BR_TAG}\n')
    markdown_lines.append('\n')

    markdown_lines.append(f'{BEGIN_MERMAID_SECTION}\n')
    markdown_lines.append(f'{BEGIN_GRAPH}\n')
    quantized_nodes = []
    for edge in edges:
        current_nodes = []
        for node_id in edge: # edge = (begin_node_id, end_node_id)
            node_name = ', '.join(_op_type_to_list(node_id_to_node_data[node_id].get('type', 'Const')))
            current_nodes.append(f'{node_id}({node_name})')
            if 'input' in node_id or 'output' in node_id:
                quantized_nodes.append(node_id)
        edge_as_str = ' --> '.join(current_nodes)
        markdown_lines.append(f'{SHIFT}{edge_as_str}\n')

    for node_id in quantized_nodes:
        markdown_lines.append(f'{SHIFT}style {node_id} fill:{QUANTIZED_NODE_COLOR}\n')

    markdown_lines.append(f'{END_MERMAID_SECTION}\n')
    markdown_lines.append('\n')
    markdown_lines.append(f'{DELIMITER}\n')
    markdown_lines.append('\n')
    return markdown_lines


def _dump_patterns_to_markdown(ignored_patterns):
    patterns = [p for pattern_group in ignored_patterns.values() for p in pattern_group]
    patterns.sort(key=lambda pattern: pattern['name'])
    markdown_lines = []
    for pattern in patterns:
        markdown_lines.extend(dump_pattern_to_mermaid_flowchart(pattern))
    return markdown_lines


def _extract_patterns_from_markdown(path_to_markdown):
    BEGIN_IGNORED_PATTERNS_SECTION = '<!--IGNORED PATTERNS-->'
    with open(path_to_markdown) as md:
        markdown_lines = md.readlines()
    start_idx = markdown_lines.index(f'{BEGIN_IGNORED_PATTERNS_SECTION}\n') + 1
    target_lines = markdown_lines[start_idx:]
    return list(filter(lambda line: 'Examples' not in line, target_lines))


def test_ignored_patterns_documentation():
    # Skipped due to the need to update the behavior of templates
    pytest.skip()
    IGNORED_PATTERNS_DOC_PATH = Path(hardware.__file__).parent.absolute() / 'IgnoredPatterns.md'
    expected = _dump_patterns_to_markdown(get_ignored_patterns())
    actual = _extract_patterns_from_markdown(IGNORED_PATTERNS_DOC_PATH)
    assert expected == actual
