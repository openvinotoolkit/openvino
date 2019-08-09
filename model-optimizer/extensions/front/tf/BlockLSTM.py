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

import networkx as nx

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.utils.error import Error


class BlockLSTM(FrontReplacementOp):
    """
    We prepare TensorFlow BlockLSTM op to be replaced with LSTMSequence op that will be repacked to TensorIterator later

    TensorFlow BlockLSTM op description:

        Op parameters:
         cell_clip:    Value to clip the 'cs' value to.
         use_peephole: Whether to use peephole weights.
         forget_bias:  The forget gate bias.

        Inputs:
         0: seq_len_max:  Maximum time length actually used by this input. Outputs are padded with 0s beyond this length
         1: x:            The sequence input to the LSTM, shape (timelen, batch_size, num_inputs)
         2: cs_prev:      Value of the initial cell state
         3: h_prev:       Initial output of cell (to be used for peephole)
         4: w:            The weight matrix
         5: wci:          The weight matrix for input gate peephole connection
         6: wcf:          The weight matrix for forget gate peephole connection
         7: wco:          The weight matrix for output gate peephole connection
         8: b:            The bias vector

        Outputs:
         0: i:            The input gate                    over the whole time sequence
         1: cs:           The cell state before the tanh    over the whole time sequence
         2: f:            The forget gate                   over the whole time sequence
         3: o:            The output gate                   over the whole time sequence
         4: ci:           The cell input                    over the whole time sequence
         5: co:           The cell after the tanh           over the whole time sequence
         6: h:            The output h vector               over the whole time sequence

    Limitations:
    - peephole connection, so we check `use_peephole`!=True and cut `wci`, `wco`, `wcf` off
    - cell_clip parameter, so we check `cell_clip==-1`, which means we do not clip
    """
    op = "BlockLSTM"
    enabled = True

    def nodes_to_remove(self, graph: Graph, match: dict):
        # do not remove matched node
        return []

    @staticmethod
    def find_key_by_input_port(u: Node, v: Node, p: int):
        key = None
        for k, edge_info in u.graph.get_edge_data(u.id, v.id).items():
            if p == edge_info['in']:
                return k
        return key

    def run_after(self):
        from extensions.front.restore_ports import RestorePorts
        return [RestorePorts]

    def replace_op(self, graph: Graph, node: Node):
        if node.use_peephole:
            raise Error("BlockLSTM operation is not supported with `use_peephole`==True. Node: {}"
                        "".format(node.soft_get('name')))

        if node.cell_clip != -1:
            raise Error("Clipping is not supported for BlockLSTM operation. `cell_clip`={!s} for node: {}"
                        "".format(node.cell_clip, node.soft_get('name')))

        log.debug("Start BlockLSTM->LSTMSequence translation for node: {} with parameters:\n"
                  "`cell_clip`={!s}, `use_peephole`=={!s}, `forget_bias`={!s}\n"
                  "inputs: {},\noutputs:{}".format(node.soft_get('name'), node.cell_clip, node.use_peephole,
                                                   node.forget_bias, {p: i.id for p, i in node.in_nodes().items()},
                                                   {p: o.id for p, o in node.out_nodes().items()}))

        log.debug("Cutting all inputs for peephole connection (5, 6, 7 input ports) off, as `use_peephole`=False")

        for p, input_data in node.in_nodes().items():
            if p in [5, 6, 7]:
                key = self.find_key_by_input_port(node.in_node(p), node, p)
                assert key is not None
                graph.remove_edge(node.in_node(p).id, node.id, key=key)

        log.debug("Cutting seq_len_max input off")
        graph.remove_edge(node.in_node(0).id, node.id)

        """
        Reconnecting input edges of LSTMSequence:
        TF input edges:             Description:                 MO input edges:
              1                          input                        0
              4                         weights                       1
              8                         biases                        2
              3               h_prev: initial output of cell          3
              2               cs_prev: initial cell state             4
        """
        inputs = node.in_edges()
        assert 1 in inputs, "Sequence input to the BlockLSTM is required (1 port). Node {}".format(node.id)
        assert 2 in inputs, "Value of the initial cell state is required (2 port). Node {}".format(node.id)
        assert 3 in inputs, "Initial output of cell is required input to BlockLSTM (3 port). Node {}".format(node.id)
        assert 4 in inputs, "The weight matrix is required input to BlockLSTM (4 port) . Node {}".format(node.id)
        assert 8 in inputs, "The bias vector is required input to BlockLSTM (8 port). Node {}".format(node.id)

        inputs[3]['in'] = 3
        inputs[1]['in'] = 0
        inputs[4]['in'] = 1
        inputs[2]['in'] = 4
        inputs[8]['in'] = 2

        log.debug("Checking for unsupported outputs usage (output ports: 0, 2, 3, 4, 5)")
        for port, input_data in node.out_nodes().items():
            if port in [0, 2, 3, 4, 5]:
                raise Error("Output port {} of BlockLSTM node {} is not supported".format(node.id, port))

        """
        Reconnecting output edges of LSTMSequence:
        TF output edges:             Description:                 MO output edges:
              6                     output h vector                     0
              1                   cell state before the tanh            1
        """

        outputs = node.out_edges()
        if 6 in outputs:
            outputs[6]['out'] = 0
            node.add_output_port(0, skip_if_exist=True)

        # do not replace any output edge
        return []
