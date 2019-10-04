"""
 Copyright (c) 2019 Intel Corporation

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

import networkx as nx
import numpy as np

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class Unique(Op):
    ''' The operation finds unique elements in 1-D tensor.
        For more details see https://www.tensorflow.org/api_docs/python/tf/unique

        attributes:
            - sorted, indicates whether to sort the unique elements in ascending order or
                      to return in the same order as they occur in the input
            - return_inverse, indicates whether to output indices
            - return_counts, indicates whether to output the counts of each unique element

        1 input:
            - [0, required] input tensor (1D)
        
        2 outputs:
            - [0, required] tensor containing all of the unique elements of the input
                            and sorted in the same order as in the input (1D)
            - [1, optional] tensor of indices for each value of the input
                            in the tensor of unique elements (1D)
            - [2, optional] tensor with a number of occurences for each unique element
                            in the input (1D)
    '''
    op = 'Unique'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'infer': __class__.infer,
            'in_ports_count': 1,
            'out_ports_count': 3
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'sorted',
            'return_inverse',
            'return_counts',
        ]

    @staticmethod
    def infer(node: Node):
        # check that all required attributes are set
        assert node.has('sorted') and node.sorted in ['true', 'false'], \
            "Unique does not have valid sorted attribute"
        assert node.has('return_inverse') and node.return_inverse in ['true', 'false'], \
            "Unique does not have valid return_inverse attribute"
        assert node.has('return_counts') and node.return_counts in ['true', 'false'], \
            "Unique does not have valid return_counts attribute"

        # check a number of input and output nodes
        assert len(node.in_nodes()) == 1, "Unique must have one input"
        assert len(node.out_nodes()) <= 3, "Unique must have less or equal to 3 outputs"

        # compute maximum number of outputs if no output port is pruned
        max_num_outputs = 1
        if node.return_inverse == 'true':
            max_num_outputs += 1
        if node.return_counts == 'true':
            max_num_outputs += 1

        # check a number of outputs
        assert len(node.out_nodes()) <= max_num_outputs, \
            "The number of outputs in IR Unique layer must be less or equal to framework graph one"
        
        # check that the output with unique elements remains in a graph after pruning
        # since this is required output
        assert 0 in node.out_nodes(), \
            "The output with unique elements must remain in a graph"

        # check if outputs with indices and counts remain in a graph after pruning
        # and update attributes
        if len(node.out_nodes()) == 1:
            node.return_inverse = 'false'
            node.return_counts = 'false'
        if len(node.out_nodes()) == 2 and 1 in node.out_nodes() \
        and node.return_inverse == 'true' and node.return_counts == 'true':
            node.return_counts = 'false'
        if len(node.out_nodes()) == 2 and 2 in node.out_nodes() \
        and node.return_inverse == 'true' and node.return_counts == 'true':
            node.return_inverse = 'false'

        # check that input is 1-D tensor
        input_shape = node.in_node(0).shape
        assert input_shape is not None and input_shape.size == 1, \
            "Unique accepts only 1-D input"

        # determine a shape for each output
        for out_node_ind in node.out_nodes():
            assert (out_node_ind < max_num_outputs), "Unique has three outputs at most"
            # all outputs have the same shape equal to the input shape
            node.out_node(out_node_ind).shape = input_shape

        input_value = node.in_node(0).value
        if input_value is None:
            return

        # check that input value is 1-D
        assert len(input_value.shape) == 1, \
            "Unique accepts only 1-D input"

        is_sorted = (node.sorted == 'true')
        return_inverse = (node.return_inverse == 'true')
        return_counts = (node.return_counts == 'true')

        # infer if the input is constant
        if is_sorted:
            unique_output = np.unique(input_value, return_inverse = return_inverse,
                                      return_counts = return_counts, return_index = False)
            if not return_inverse and not return_counts:
                unique_output = [unique_output]
        else:
            # np.unique can only return unique elements in sorted order
            # so this case should be handled separately
            sorted_uniques, sorted_index, sorted_inverse, sorted_counts = np.unique(input_value, return_index = True,
                                                               return_inverse = True, return_counts = True)
            # compute uniques that are in the same order as they occur in the input,
            # indices of input values in uniques, counts for each unique element
            uniques = []
            inverse = []
            counts = []
            old_ind_by_elem = dict(zip(sorted_uniques, range(len(sorted_index))))
            new_ind_by_elem = dict()
            new_ind = 0
            for ind in np.sort(sorted_index):
                uniques.append(input_value[ind])
                old_ind = old_ind_by_elem[input_value[ind]]
                counts.append(sorted_counts[old_ind])
                new_ind_by_elem[input_value[ind]] = new_ind
                new_ind += 1
            inverse = [new_ind_by_elem[input_value[ind]] for ind in range(len(input_value))]

            # pack unique_output
            unique_output = []
            unique_output.append(uniques)
            if return_inverse:
                unique_output.append(inverse)
            if return_counts:
                unique_output.append(counts)

        # write result to output nodes
        j = 0
        for out_node_ind in node.out_nodes():
            node.out_node(out_node_ind).value = np.array(unique_output[j], dtype=np.float)
            node.out_node(out_node_ind).shape = np.array(node.out_node(out_node_ind).value.shape, dtype=np.int64)
            j += 1
