# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import copy

from extensions.back.ConvolutionNormalizer import ConvolutionNormalizer, ConvolutionWithGroupsResolver
from extensions.back.PackBinaryWeights import PackBinaryWeights
from extensions.back.SpecialNodesFinalization import RemoveConstOps, CreateConstNodesReplacement
from extensions.back.StridedSliceMasksNormalizer import StridedSliceMasksNormalizer
from extensions.back.blob_normalizer import BlobNormalizer
from extensions.back.kaldi_remove_memory_output import KaldiRemoveMemoryOutputBackReplacementPattern
from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import data_type_str_to_precision
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from mo.pipeline.common import prepare_emit_ir
from mo.utils.class_registration import apply_replacements_list
from mo.utils.ir_engine.ir_engine import IREngine
from mo.utils.ir_reader.layer_to_class import copy_graph_with_ops, collect_extenders, collect_ops
from mo.utils.utils import get_mo_root_dir
from extensions.back.MarkNodesWithShapeValues import MarkNodesWithShapeValues


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
        StridedSliceMasksNormalizer,
        PackBinaryWeights,
        BlobNormalizer,
        ConvolutionNormalizer,
        KaldiRemoveMemoryOutputBackReplacementPattern,
        MarkNodesWithShapeValues,
    ]

    # We need to run some specific passes from MO back stage.
    apply_replacements_list(graph, transformation_list)

    # Transformations with enabled=False should be run manually.
    for_graph_and_each_sub_graph_recursively(graph, RemoveConstOps().find_and_replace_pattern)
    for_graph_and_each_sub_graph_recursively(graph, CreateConstNodesReplacement().find_and_replace_pattern)

    prepare_emit_ir(graph, precision, path, name, meta_info=meta_data)
