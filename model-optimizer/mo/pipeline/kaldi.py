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
import logging as log
import numpy as np

from extensions.back.kaldi_remove_memory_output import KaldiRemoveMemoryOutputBackReplacementPattern
from extensions.back.remove_last_softmax_pattern import RemoveLastSoftMaxPattern
from extensions.front.kaldi.eliminate_redundant_reshape import EliminateRedundantReshape
from extensions.front.kaldi.fuse_repeated_reshape import FuseRepeatedReshapes
from extensions.middle.EltwiseChecker import EltwiseChecker
from mo.front.common.register_custom_ops import update_extractors_with_extensions
from mo.front.extractor import create_tensor_nodes, extract_node_attrs, add_output_ops, remove_output_ops
from mo.front.kaldi import loader
from mo.front.kaldi.extractor import kaldi_extractor, kaldi_type_extractors
from mo.utils import class_registration
from mo.utils.cli_parser import get_meta_info
from mo.utils.error import Error
from mo.utils.find_inputs import find_outputs

from mo.graph.graph import print_graph_stat, Node, check_empty_graph
from mo.middle.passes.eliminate import graph_clean_up
from mo.middle.passes.infer import override_placeholder_shapes, partial_infer, mark_outputs, override_batch
from mo.pipeline.common import prepare_emit_ir
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
        biases_node.value = np.subtract(biases_node.value, counts)
    else:
        biases_node.value = counts * -1
        biases_node.shape = counts.shape


def driver(argv, input_model, output_model_name, outputs, output_dir, scale, placeholder_shapes=None,
           mean_scale_values=()):
    meta_info = get_meta_info(argv)

    EltwiseChecker.enabled = False
    
    try:
        graph, input_shapes = loader.load_kaldi_nnet_model(input_model)
    except Exception as e:
        raise Error('Model Optimizer is not able to read Kaldi model {}. '.format(input_model) +
                    refer_to_faq_msg(91)) from e
    check_empty_graph(graph, 'load_kaldi_nnet_model')

    graph.graph['cmd_params'] = argv
    graph.graph['fw'] = 'kaldi'
    graph.graph['ir_version'] = 2 if argv.generate_deprecated_IR_V2 else 3

    if argv.counts:
        try:
            counts = loader.read_counts_file(argv.counts)
        except Exception as e:
            raise Error('Model Optimizer is not able to read counts file {}'.format(argv.counts) +
                        refer_to_faq_msg(92)) from e

    update_extractors_with_extensions(kaldi_type_extractors)
    # Intentionally before extracting attributes

    class_registration.apply_replacements(graph, class_registration.ClassType.FRONT_REPLACER)
    extract_node_attrs(graph, lambda node: kaldi_extractor(node))

    output_op_nodes = add_output_ops(graph, outputs)  # TODO pass real outputs instead of None
    log.debug("After adding specific nodes for outputs")
    print_graph_stat(graph)
    check_empty_graph(graph, 'add_output_ops')
    create_tensor_nodes(graph)

    graph_clean_up(graph)
    log.debug("After removing specific nodes for output")
    print_graph_stat(graph)

    override_placeholder_shapes(graph, placeholder_shapes)
    override_batch(graph, argv.batch)

    graph_clean_up(graph)
    log.debug("After setting input shapes")
    print_graph_stat(graph)
    graph_clean_up(graph)
    remove_output_ops(graph)
    log.debug("After removing specific nodes for output")
    print_graph_stat(graph)

    # You need to pass required network outputs here
    # but we don't have a way yet, so just passing all discovered sinks
    mark_outputs(graph)

    graph_clean_up(graph)
    log.debug("After graph_cleanup")
    print_graph_stat(graph)
    graph = partial_infer(graph)
    # The order is intentional, firstly eliminate repeated, then remove redundant
    FuseRepeatedReshapes().find_and_replace_pattern(graph)
    EliminateRedundantReshape().find_and_replace_pattern(graph)
    check_empty_graph(graph, 'partial_infer')
    if argv.counts:
        apply_biases_to_last_layer(graph, counts)

    if argv.remove_output_softmax:
        RemoveLastSoftMaxPattern().find_and_replace_pattern(graph)
        graph_clean_up(graph)
        log.debug("After removing softmax")
        print_graph_stat(graph)

    # Intentionally after all transformations
    KaldiRemoveMemoryOutputBackReplacementPattern().find_and_replace_pattern(graph)
    prepare_emit_ir(graph, argv.data_type, output_dir, output_model_name, meta_info=meta_info)
    return 0
