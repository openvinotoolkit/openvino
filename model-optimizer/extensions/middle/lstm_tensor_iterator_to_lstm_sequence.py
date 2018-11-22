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

import networkx as nx

from mo.graph.graph import copy_node
from mo.utils.error import Error
from mo.middle.pattern_match import find_isomorphisms
from mo.middle.replacement import MiddleReplacementPattern
from extensions.ops.lstm_sequence import LSTMSequence
from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
from extensions.middle.lstm_sequence_normalize import LSTMSequenceNormalize, permute_before_and_after
from extensions.middle.lstm_sequence_tensor_iterator import LSTMSequenceTensorIterator
from extensions.middle.TF_lstm_cell_to_generic import TensorFlowLSTMtoGeneric


class TensorIteratorLSTM(MiddleReplacementPattern):
    """ Detects TensorIterator with LSTMCell of supported form.

        Collect original operation names of supported LSTMCells in
        the list LSTMCell.instances_supported_by_IE. It will be used at the second
        round of the network translation. Mark all supported LSTMCell with flag
        supported_by_IE to have a chance to detect all not-supported instances
        in a separate pass.
    """

    enabled = True

    def run_after(self):
        return [TensorIteratorMerge, LSTMSequenceNormalize, LSTMSequenceTensorIterator, TensorFlowLSTMtoGeneric]

    def pattern(self):
        return dict(
            nodes=[
                ('ti', dict(kind='op', op='TensorIterator')),
            ],
            edges=[
            ]
        )

    @staticmethod
    def replace_pattern(graph: nx.MultiDiGraph, match: dict):
        nodes=[
            ('input_unsqueezed'),
            ('squeeze', dict(op='Reshape')),
            ('input_squeezed'),
            ('input_hidden'),
            ('input_cell'),
            ('weights'),
            ('biases'),

            ('lstm', dict(op='LSTMCell')),

            ('output_hidden'),
            ('output_cell'),
            ('unsqueeze', dict(op='Reshape')),
            ('output_unsqueezed'),
        ]
        edges=[
            ('input_unsqueezed', 'squeeze'),
            ('squeeze', 'input_squeezed'),

            ('input_squeezed', 'lstm', {'in': 0}),
            ('input_hidden', 'lstm', {'in': 1}),
            ('input_cell', 'lstm', {'in': 2}),
            ('weights', 'lstm', {'in': 3}),
            ('biases', 'lstm', {'in': 4}),

            ('lstm', 'output_hidden', {'out': 0}),
            ('lstm', 'output_cell', {'out': 1}),

            ('output_hidden', 'unsqueeze'),
            ('unsqueeze', 'output_unsqueezed'),
        ]
        ti = match['ti']
        isomorphisms = find_isomorphisms(ti.body, nodes, edges)
        if len(list(isomorphisms)) != 1:
            raise Error('Unsupported TensorIterator layer {} was found: either its body, ports or '
                        'edges are not supported by Inference Engine. '
                        'Only TensorIterator with LSTMCell in a body of strict form is supported. '
                        'Please modify the original network '
                        'to meet the requirements.'.format(ti.soft_get('name')))
        body_match = isomorphisms[0]
        if body_match['input_hidden'].has_valid('value') or body_match['input_cell'].has_valid('value'):
            raise Error('Unsupported TensorIterator layer {} was found: initial hidden and/or cell states '
                        'for LSTMCell are constants. This is not supported. '
                        'Only TensorIterator with LSTMCell in a body of strict form is supported. '
                        'Please modify the original network '
                        'to meet the requirements.'.format(ti.soft_get('name')))
        # TODO Additional checks for port indices
        if body_match['lstm'].has_valid('mark_supported_by_IE'):
            body_match['lstm'].mark_supported_by_IE(body_match['lstm'])


class CheckUnsupportedLSTMCell(MiddleReplacementPattern):
    """ Finds all unsupported LSTMCell.

        Initiates the second translation round if find any not supported LSTMCell instances.
    """

    enabled = True

    def run_after(self):
        return [TensorIteratorLSTM]

    def pattern(self):
        return dict(
            nodes=[
                ('lstm', dict(op='LSTMCell')),
            ],
            edges=[
            ]
        )

    @staticmethod
    def replace_pattern(graph: nx.MultiDiGraph, match: dict):
        lstmcell = match['lstm']
        if lstmcell.has_valid('finalize_first_round'):
            lstmcell.finalize_first_round()
            if not lstmcell.has_and_set('supported_by_IE'):
                # this is a signal for the main translation pipeline to repeat the entire conversion process
                graph.graph['repeat_conversion'] = True
        # in case when there is no lstmcell.finalize_first_round then this cell wasn't created with the pattern
        # (for example in ONNX) and we don't initiate the second round.
