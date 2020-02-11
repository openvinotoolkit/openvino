"""
 Copyright (C) 2018-2020 Intel Corporation

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

import logging as log
from copy import copy

from extensions.back.blob_normalizer import BlobNormalizer
from extensions.back.ConvolutionNormalizer import ConvolutionNormalizer, ConvolutionWithGroupsResolver
from extensions.back.SpecialNodesFinalization import RemoveConstOps, CreateConstNodesReplacement
from extensions.back.StridedSliceMasksNormalizer import StridedSliceMasksNormalizer
from extensions.back.TopKNormalizer import TopKNormalizer

from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import np_data_type_to_precision
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from mo.pipeline.common import prepare_emit_ir

from mo.utils.ir_engine.ir_engine import IREngine
from mo.utils.ir_reader.layer_to_class import copy_graph_with_ops, collect_extenders, collect_ops
from mo.utils.utils import get_mo_root_dir


def restore_graph_from_ir(path_to_xml: str, path_to_bin: str = None) -> (Graph, dict):
    """
    Function to make valid graph and metadata for MO back stage from IR.
    :param path_to_xml:
    :param path_to_bin:
    :return: (restored graph, meta data)
    """
    ir = IREngine(path_to_xml, path_to_bin)
    assert ir.graph.graph.get('ir_version') >= 10, 'IR version {} is not supported, ' \
        'please generate actual IR for your model and use it.'.format(ir.graph.graph.get('ir_version'))

    path = get_mo_root_dir()
    collect_ops(path)
    collect_extenders(path)

    # Create a new copy of graph with correct attributes (shape & type infer, backend attrs etc.)
    new_graph = copy_graph_with_ops(ir.graph)

    return new_graph, copy(ir.meta_data)


def save_restored_graph(graph: Graph, path: str, meta_data, name=None):
    """
    Function to apply all necessary transforms from back stage to prepare and save restored graph and metadata.
    :param graph: Graph to save
    :param path: Path to saved IR
    :param meta_data: Namespace with converting parameters restored from IR
    :param name: Name for saved IR
    :return:
    """

    if name is None:
        name = graph.name

    precisions = set()

    for op in graph.get_op_nodes():
        if op.type in ('Convolution', 'MatMul'):
            if op.in_port(1).get_source().node.type == 'FakeQuantize':
                data_type = op.in_port(1).get_source().node.in_port(0).get_source().node.soft_get('data_type', None)
            else:
                data_type = op.in_port(1).get_source().node.soft_get('data_type', None)

            if data_type is not None:
                precisions.add(np_data_type_to_precision(data_type))
            else:
                log.warning('Cannot check data type for node {} with type {}, skip it.'.format(op.name, op.type))

    precision = 'FP16' if 'FP16' in precisions else 'FP32'

    # We need to run some specific passes from MO back stage.
    # After some of them we need to clean up graph!
    for_graph_and_each_sub_graph_recursively(graph, ConvolutionWithGroupsResolver().find_and_replace_pattern)
    for_graph_and_each_sub_graph_recursively(graph, TopKNormalizer().find_and_replace_pattern)
    graph.clean_up()

    for_graph_and_each_sub_graph_recursively(graph, StridedSliceMasksNormalizer().find_and_replace_pattern)

    for_graph_and_each_sub_graph_recursively(graph, BlobNormalizer().find_and_replace_pattern)
    for_graph_and_each_sub_graph_recursively(graph, ConvolutionNormalizer().find_and_replace_pattern)
    for_graph_and_each_sub_graph_recursively(graph, RemoveConstOps().find_and_replace_pattern)
    for_graph_and_each_sub_graph_recursively(graph, CreateConstNodesReplacement().find_and_replace_pattern)

    prepare_emit_ir(graph, precision, path, name, meta_info=meta_data)
