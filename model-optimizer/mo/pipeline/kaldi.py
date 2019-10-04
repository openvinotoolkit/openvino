"""
 Copyright (c) 2018-2019 Intel Corporation

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

import numpy as np

from extensions.back.CreateConstNodes import CreateConstNodesReplacement
from extensions.back.CutMemory import CutMemory
from extensions.back.ElementwiseOpsToEltwiseOps import DivideToEltwises, SubtractToEltwises, SimpleEltwiseToEltwiseOp
from extensions.back.ForceStrictPrecision import ForceStrictPrecision
from extensions.back.LeakyReluToReluWithNegativeSlope import LeakyReluToReluWithNegativeSlope
from extensions.back.ParameterToPlaceholder import ParameterToInput
from extensions.back.TransposeToPermute import TransposeToPermute
from extensions.back.kaldi_remove_memory_output import KaldiRemoveMemoryOutputBackReplacementPattern
from extensions.back.remove_last_softmax_pattern import RemoveLastSoftMaxPattern
from extensions.front.kaldi.eliminate_redundant_reshape import EliminateRedundantReshape
from extensions.front.kaldi.fuse_repeated_reshape import FuseRepeatedReshapes
from extensions.front.kaldi.replace_lstm_node_pattern import ReplaceLSTMNodePattern
from extensions.middle.EltwiseChecker import EltwiseChecker
from extensions.middle.InsertSelect import AddSelectBeforeMemoryNodePattern
from extensions.middle.RemoveDuplicationMemory import RemoveMemoryDuplicationPattern, MergeNeighborSplicePattern
from extensions.middle.RemoveIdentity import RemoveIdentity
from extensions.middle.RemoveUselessCrops import RemoveUselessCropsPattern
from extensions.middle.ReplaceMemoryOffsetWithSplice import ReplaceMemoryOffsetNodePattern, \
    ReplaceMemoryOffsetWithMemoryNodePattern
from extensions.middle.ReplacePNorm import ReplacePNormNodePattern
from extensions.middle.ReplaceSpliceNodePattern import ReplaceSpliceNodePattern
from mo.front.common.register_custom_ops import update_extractors_with_extensions
from mo.front.extractor import extract_node_attrs, remove_output_ops
from mo.front.kaldi.extractor import kaldi_extractor, kaldi_type_extractors
from mo.front.kaldi.loader.loader import load_kaldi_model, read_counts_file
from mo.graph.graph import Node
from mo.middle.passes.conv import convert_matmul_to_fully_connected
from mo.middle.passes.eliminate import graph_clean_up, remove_const_ops
from mo.middle.passes.infer import partial_infer
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from mo.pipeline.common import prepare_emit_ir
from mo.utils import class_registration
from mo.utils.cli_parser import get_meta_info
from mo.utils.error import Error
from mo.utils.find_inputs import find_outputs
from mo.utils.logger import log_step
from mo.utils.utils import refer_to_faq_msg


def apply_biases_to_last_layer(graph, counts):
    """
    The idea is the following. If the user provides counts file, it is a file that contains log-apriory probabilities,
    technically it should be subtracted from the bias of the last layer unless it is a SoftMax.
    
    Case 1:
        weights ---\
        biases  ---\
    some layer  ---> AffineTransform ---> SoftMax
    
    Then, counts are applied to biases of Affine Transform:
    
        weights             ---\
        (biases - counts)   ---\
    some layer              ---> AffineTransform ---> SoftMax
    
    Case 2:
        weights ---\
        biases  ---\
    some layer  ---> AffineTransform
    
    Just takes the last layer and updates biases:
    
        weights             ---\
        (biases - counts)   ---\
    some layer              ---> AffineTransform
    
    Parameters
    ----------
    graph
    counts

    Returns
    -------

    """""
    outputs_ids = find_outputs(graph)
    for output in outputs_ids.copy():
        node = Node(graph, output)
        if node.in_node().op != 'Memory':
            continue
        outputs_ids.remove(output)

    if len(outputs_ids) > 1:
        raise Error('Ambiguity in applying counts to several outputs.')
    elif len(outputs_ids) == 0:
        raise Error('No outputs were found')

    node = Node(graph, outputs_ids[0])
    target_node = node.in_node()
    if target_node and target_node['op'] == 'SoftMax':
        data_node = target_node.in_node()
        target_node = data_node.in_node()

    biases_node = target_node.in_nodes()[2]  # first - input, second - weights, third - biases
    if biases_node.value is not None:
        biases_node.value = np.subtract(biases_node.value, counts)  # pylint: disable=assignment-from-no-return
    else:
        biases_node.value = counts * -1
        biases_node.shape = counts.shape


def driver(argv, input_model, output_model_name, output_dir):
    log_step(argv.steps, 'LOAD')
    meta_info = get_meta_info(argv)

    EltwiseChecker.enabled = False

    try:
        graph = load_kaldi_model(input_model)
    except Exception as e:
        raise Error('Model Optimizer is not able to parse Kaldi model {}. '.format(input_model) +
                    refer_to_faq_msg(91)) from e
    graph.check_empty_graph('load_kaldi_nnet_model')
    graph.graph['cmd_params'] = argv
    graph.graph['fw'] = 'kaldi'

    if graph.graph['cmd_params'].generate_experimental_IR_V10:
        version = 10
    else:
        version = 6
    graph.graph['ir_version'] = 2 if argv.generate_deprecated_IR_V2 else version

    update_extractors_with_extensions(kaldi_type_extractors)
    extract_node_attrs(graph, lambda node: kaldi_extractor(node))

    # --------------------------------- LOAD END ------------------------------------------------------
    log_step(argv.steps, 'FRONT')
    ReplaceLSTMNodePattern().find_and_replace_pattern(graph)
    class_registration.apply_replacements(graph, class_registration.ClassType.FRONT_REPLACER)
    log_step(argv.steps, 'MIDDLE')
    graph = partial_infer(graph)

    ReplacePNormNodePattern().find_and_replace_pattern(graph)
    ReplaceMemoryOffsetNodePattern().find_and_replace_pattern(graph)
    ReplaceMemoryOffsetWithMemoryNodePattern().find_and_replace_pattern(graph)
    RemoveMemoryDuplicationPattern().find_and_replace_pattern(graph)
    MergeNeighborSplicePattern().find_and_replace_pattern(graph)
    RemoveUselessCropsPattern().find_and_replace_pattern(graph)
    RemoveIdentity().find_and_replace_pattern(graph)
    graph_clean_up(graph)

    AddSelectBeforeMemoryNodePattern().find_and_replace_pattern(graph)

    ReplaceSpliceNodePattern().find_and_replace_pattern(graph)
    graph_clean_up(graph)

    # The order is intentional, firstly eliminate repeated, then remove redundant
    FuseRepeatedReshapes().find_and_replace_pattern(graph)
    EliminateRedundantReshape().find_and_replace_pattern(graph)
    graph_clean_up(graph)
    graph.check_empty_graph('partial_infer')
    if argv.counts:
        try:
            counts = read_counts_file(argv.counts)
        except Exception as e:
            raise Error('Model Optimizer is not able to read counts file {}'.format(argv.counts) +
                        refer_to_faq_msg(92)) from e

        apply_biases_to_last_layer(graph, counts)

    if argv.remove_output_softmax:
        RemoveLastSoftMaxPattern().find_and_replace_pattern(graph)
        graph_clean_up(graph)
        log.debug("After removing softmax")
        graph.print_graph_stat()

    log_step(argv.steps, 'BACK')
    LeakyReluToReluWithNegativeSlope().find_and_replace_pattern(graph)
    TransposeToPermute().find_and_replace_pattern(graph)
    DivideToEltwises().find_and_replace_pattern(graph)
    SubtractToEltwises().find_and_replace_pattern(graph)
    SimpleEltwiseToEltwiseOp().find_and_replace_pattern(graph)
    for_graph_and_each_sub_graph_recursively(graph, convert_matmul_to_fully_connected)

    # Intentionally after all transformations
    if argv.remove_memory:
        CutMemory().find_and_replace_pattern(graph)
        graph_clean_up(graph)
    ParameterToInput().find_and_replace_pattern(graph)

    KaldiRemoveMemoryOutputBackReplacementPattern().find_and_replace_pattern(graph)
    ForceStrictPrecision().find_and_replace_pattern(graph)
    remove_const_ops(graph)
    CreateConstNodesReplacement().find_and_replace_pattern(graph)

    remove_output_ops(graph)
    log_step(argv.steps, 'EMIT')
    prepare_emit_ir(graph, argv.data_type, output_dir, output_model_name, meta_info=meta_info)
    return 0
