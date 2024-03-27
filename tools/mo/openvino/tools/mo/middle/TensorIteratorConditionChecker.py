# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import compatible_dims
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class ConditionChecks(MiddleReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['is_cyclic']]

    def run_after(self):
        from openvino.tools.mo.middle.TensorIteratorBackEdge import BackEdgesMatching
        return [BackEdgesMatching]

    def run_before(self):
        from openvino.tools.mo.middle.TensorIteratorMerge import TensorIteratorMerge
        return [TensorIteratorMerge]

    @staticmethod
    def pattern():
        log.debug('+++++++++++++++ ConditionCheckerMatching ++++++++++++++++')
        return dict(
            nodes=[
                ('condition', dict(kind='op', op='TensorIteratorCondition')),
                ('Strided_slice', dict(kind='op', op='StridedSlice')),
                ('Strided_slice_data', dict(kind='data')),
                ('shape', dict(kind='op', op='ShapeOf')),
                ('shape_data', dict(kind='data')),

                ('minimum', dict(kind='op', op='Minimum')),
                ('minimum_data', dict(kind='data')),
                ('Maximum', dict(kind='op', op='Maximum')),
                ('Maximum_data', dict(kind='data')),
            ],
            edges=[
                ('shape', 'shape_data'),
                ('shape_data', 'Strided_slice'),
                ('Strided_slice', 'Strided_slice_data'),
                ('Strided_slice_data', 'condition'),
                ('Strided_slice_data', 'minimum'),

                ('Maximum', 'Maximum_data'),
                ('Maximum_data', 'minimum'),
                ('minimum', 'minimum_data'),
                ('minimum_data', 'condition'),
            ],
        )

    @staticmethod
    def replace_pattern(graph, match: dict):
        # Check for SS params
        # Sanity check that we iterate over axis of some tensor
        ss = match['Strided_slice']
        params = ss.in_nodes()
        assert np.all(params[1].in_node().value == 0)
        assert np.all(params[2].in_node().value == 1)
        assert np.all(params[3].in_node().value == 1)

        # Check for comparing SS and seq_length source (it should be one tensor)
        # SIMPLE CHECK
        assert match['Strided_slice_data'].value is not None
        if match['minimum_data'].value is None:
            log.warning('TF loop doesn\'t have a constant upper bound produced by node {}, or ModelOptimizer '
                        'cannot detect a constant in this case. Loops with a dynamic number of iterations are not '
                        'supported, so in the resulting IR, generated TensorIterator will have '
                        'a maximum number of iterations determined by input tensor size: {}'
                        ''.format(match['minimum_data'].soft_get('name'), match['Strided_slice_data'].value)
                        )
        else:
            assert compatible_dims(match['Strided_slice_data'].value, match['minimum_data'].value), \
                'Values do not match: {} and {}'.format(match['Strided_slice_data'].value, match['minimum_data'].value)

        # Check that bound for Condition and Inputs/Outputs sizes match
        condition_time = match['condition'].out_node(0)
        inputs_and_outputs = condition_time.out_nodes()
        type_list = ['TensorIteratorInput']

        for ta in inputs_and_outputs:
            if ta.has_valid('kind') and ta['kind'] == 'op' and ta['op'] in type_list:
                assert ta.in_node(0).id == ss.id

        log.debug('+++++++++++++++ Condition Check was successful ++++++++++++++++')
