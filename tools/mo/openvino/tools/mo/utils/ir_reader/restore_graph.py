# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from copy import copy

from openvino.tools.mo.back.ConvolutionNormalizer import ConvolutionNormalizer, ConvolutionWithGroupsResolver
from openvino.tools.mo.back.ShapeOfConstFolding import ShapeOfConstFolding
from openvino.tools.mo.back.MarkNodesWithShapeValues import MarkNodesWithShapeValues
from openvino.tools.mo.back.PackBinaryWeights import PackBinaryWeights
from openvino.tools.mo.back.SpecialNodesFinalization import RemoveConstOps, CreateConstNodesReplacement
from openvino.tools.mo.back.StridedSliceMasksNormalizer import StridedSliceMasksNormalizer
from openvino.tools.mo.back.blob_normalizer import BlobNormalizer
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.passes.convert_data_type import data_type_str_to_precision
from openvino.tools.mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from openvino.tools.mo.pipeline.common import prepare_emit_ir
from openvino.tools.mo.utils.class_registration import apply_replacements_list
from openvino.tools.mo.utils.ir_engine.ir_engine import IREngine
from openvino.tools.mo.utils.ir_reader.layer_to_class import copy_graph_with_ops, collect_extenders, collect_ops
from openvino.tools.mo.utils.utils import get_mo_root_dir


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


def save_restored_graph(graph: Graph, path: str, meta_data, name=None, rename_results=True):
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

    if 'data_type' not in meta_data:
        log.debug('Provided `meta_data` does not contain `data_type` parameter. Set `data_type`'
                  ' parameter value to `FP32`.')
        # Set data_type to FP32. All restored constants will be saved in provided data type.
        data_type = 'FP32'

        # We need to specify this attribute to pass graph transformations. This information will not be saved into IR.
        # All constants and placeholders will be saved with same types as restored from IR
        graph.graph['cmd_params'].data_type = data_type
    else:
        data_type = data_type_str_to_precision(graph.graph['cmd_params'].data_type)

    assert data_type in ['FP16', 'FP32'], '`data_type` value {} is not supported by MO,' \
                                          ' cannot save graph'.format(data_type)

    # List items order matters, do not change it.
    transformation_list = [
        ConvolutionWithGroupsResolver,
        ShapeOfConstFolding,
        StridedSliceMasksNormalizer,
        PackBinaryWeights,
        BlobNormalizer,
        ConvolutionNormalizer,
        MarkNodesWithShapeValues,
    ]

    # We need to run some specific passes from MO back stage.
    apply_replacements_list(graph, transformation_list)

    # Transformations with enabled=False should be run manually.
    for_graph_and_each_sub_graph_recursively(graph, RemoveConstOps().find_and_replace_pattern)
    for_graph_and_each_sub_graph_recursively(graph, CreateConstNodesReplacement().find_and_replace_pattern)

    prepare_emit_ir(graph, data_type, path, name, meta_info=meta_data, rename_results=rename_results)
