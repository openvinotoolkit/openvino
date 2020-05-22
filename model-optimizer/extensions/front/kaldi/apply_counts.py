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
import numpy as np

from extensions.ops.elementwise import Add
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Node, Graph
from mo.utils.error import Error
from mo.utils.find_inputs import find_outputs
from mo.utils.utils import refer_to_faq_msg


def apply_biases_to_last_layer(graph, counts):
    """
    When user provides counts file, it is a file that contains log-apriory probabilities,
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
        if node.op != 'Assign':
            continue
        outputs_ids.remove(output)

    if len(outputs_ids) > 1:
        raise Error('Ambiguity in applying counts to several outputs.')
    elif len(outputs_ids) == 0:
        raise Error('No outputs were found')

    target_node = Node(graph, outputs_ids[0])
    if target_node.op == 'SoftMax':
        target_node = target_node.in_port(0).get_source().node

    sub_node = create_op_node_with_second_input(graph, Add, -counts, {'name': 'sub_counts'})
    target_node.out_port(0).get_connection().set_source(sub_node.out_port(0))
    sub_node.in_port(0).connect(target_node.out_port(0))


def read_counts_file(file_path):
    with open(file_path, 'r') as f:
        file_content = f.readlines()
    if len(file_content) > 1:
        raise Error('Expect counts file to be one-line file. ' +
                    refer_to_faq_msg(90))

    counts_line = file_content[0].strip().replace('[', '').replace(']', '')
    try:
        counts = np.fromstring(counts_line, dtype=float, sep=' ')
    except TypeError:
        raise Error('Expect counts file to contain list of floats.' +
                    refer_to_faq_msg(90))
    cutoff = 1.00000001e-10
    cutoff_idxs = np.where(counts < cutoff)
    counts[cutoff_idxs] = cutoff
    scale = 1.0 / np.sum(counts)
    counts = np.log(counts * scale)  # pylint: disable=assignment-from-no-return
    counts[cutoff_idxs] += np.finfo(np.float32).max / 2
    return counts


class ApplyCountsFilePattern(FrontReplacementSubgraph):
    """
    Pass applies counts file as biases to last layer
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].counts is not None]

    def run_after(self):
        from extensions.front.output_cut import OutputCut
        from extensions.front.MoveEmbeddedInputsToInputs import MoveEmbeddedInputsToInputs
        return [MoveEmbeddedInputsToInputs,
                OutputCut,
                ]

    def run_before(self):
        from extensions.front.MatMul_normalizer import FullyConnectedDecomposer
        return [FullyConnectedDecomposer,
                ]

    def find_and_replace_pattern(self, graph: Graph):
        try:
            counts = read_counts_file(graph.graph['cmd_params'].counts)
        except Exception as e:
            raise Error('Model Optimizer is not able to read counts file {}'.format(graph.graph['cmd_params'].counts) +
                        refer_to_faq_msg(92)) from e

        apply_biases_to_last_layer(graph, counts)
