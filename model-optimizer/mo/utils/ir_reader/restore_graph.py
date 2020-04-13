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

from copy import copy

from extensions.back.ConvolutionNormalizer import ConvolutionNormalizer, ConvolutionWithGroupsResolver
from extensions.back.PackBinaryWeights import PackBinaryWeights
from extensions.back.SpecialNodesFinalization import RemoveConstOps, CreateConstNodesReplacement
from extensions.back.StridedSliceMasksNormalizer import StridedSliceMasksNormalizer
from extensions.back.TopKNormalizer import TopKNormalizer
from extensions.back.blob_normalizer import BlobNormalizer
from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import data_type_str_to_precision
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from mo.pipeline.common import prepare_emit_ir
from mo.utils.class_registration import apply_replacements_list
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

    precision = data_type_str_to_precision(graph.graph['cmd_params'].data_type)
    assert precision in ['FP16', 'FP32'], 'Cannot define precision for restored model!'

    # List items order matters, do not change it.
    transformation_list = [
        ConvolutionWithGroupsResolver,
        TopKNormalizer,
        StridedSliceMasksNormalizer,
        PackBinaryWeights,
        BlobNormalizer,
        ConvolutionNormalizer,
    ]

    # We need to run some specific passes from MO back stage.
    apply_replacements_list(graph, transformation_list)

    # Transformations with enabled=False should be run manually.
    for_graph_and_each_sub_graph_recursively(graph, RemoveConstOps().find_and_replace_pattern)
    for_graph_and_each_sub_graph_recursively(graph, CreateConstNodesReplacement().find_and_replace_pattern)

    prepare_emit_ir(graph, precision, path, name, meta_info=meta_data)
