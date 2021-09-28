# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from copy import copy

import numpy as np

from extensions.back.ConvolutionNormalizer import ConvolutionNormalizer, ConvolutionWithGroupsResolver
from extensions.back.MarkNodesWithShapeValues import MarkNodesWithShapeValues
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


def define_data_type(graph: Graph):
    # Trying to find parameters or constants with FP16 data_type
    for op_type in ('Parameter', 'Const'):
        for node in graph.get_op_nodes(op=op_type):
            if node.soft_get('element_type') == 'f16' or node.soft_get('data_type') == np.float16:
                log.debug('Found operation with `FP16` data type. Set graph `data_type` '
                            'attribute value to `FP16`!')
                return 'FP16'
    # If there are no parameters or constants with FP16 we return FP32 as data_type value
    log.debug('Operations with `FP16` data type not found. Set graph `data_type` '
              'attribute value to `FP32`')
    return 'FP32'


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

    if 'data_type' not in meta_data:
        log.debug('Provided graph does not contain `data_type` parameter in `meta_info` section! Trying to '
                    'define `data_type` parameter value from the model.')
        data_type = define_data_type(graph)

        # We need to specify this attribute to pass graph transformations. This information will not be saved into IR.
        graph.graph['cmd_params'].data_type = data_type
    else:
        data_type = data_type_str_to_precision(graph.graph['cmd_params'].data_type)

    assert data_type in ['FP16', 'FP32'], '`data_type` value {} is not supported by MO,' \
                                          ' cannot save graph'.format(data_type)

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

    prepare_emit_ir(graph, data_type, path, name, meta_info=meta_data)
