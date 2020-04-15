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
import unittest

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.ops.op import PermuteAttrs
from mo.ops.strided_slice import extend_mask_according_ellipsis, permute_masks, permute_array_with_ellipsis, \
    StridedSlice
from mo.utils.error import Error
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'input': {
        'kind': 'op',
        'op': None
    },
    'data_1': {
        'kind': 'data',
        'shape': None,
        'value': None,
    },
    'begin': {
        'kind': 'op',
        'op': 'Const',
        'value': None,
        'shape': None
    },
    'begin_data': {
        'kind': 'data',
        'shape': None,
        'value': np.array([]),
    },
    'end': {
        'kind': 'op',
        'op': 'Const',
        'value': None,
        'shape': None
    },
    'end_data': {
        'kind': 'data',
        'shape': None,
        'value': np.array([]),
    },
    'stride': {
        'kind': 'op',
        'op': 'Const',
        'value': None,
        'shape': None
    },
    'stride_data': {
        'kind': 'data',
        'shape': None,
        'value': np.array([]),
    },
    'strided_slice': {
        'op': 'StridedSlice',
        'begin_mask': None,
        'end_mask': None,
        'new_axis_mask': None,
        'shrink_axis_mask': None,
        'ellipsis_mask': None,
        'kind': 'op',
    },
    'data_2': {
        'kind': 'data',
        'shape': None,
        'value': None,
    }
}


class TestPermutationStridedSlice(unittest.TestCase):
    def test_permute_begin_end(self):
        # Testing constant path case
        graph = build_graph(nodes_attributes,
                            [('data_1', 'strided_slice'),
                             ('begin', 'strided_slice'),
                             ('end', 'strided_slice'),
                             ('stride', 'strided_slice'),
                             ('strided_slice', 'data_2')],
                            {'data_1': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             'strided_slice': {'begin_mask': np.array([1, 1, 0, 0]), 'end_mask': np.array([0, 1, 0, 0]),
                                               'new_axis_mask': np.array([0, 0, 0]), 'shrink_axis_mask': [0, 0, 0],
                                               'ellipsis_mask': np.array([0, 0, 0])},
                             'data_2': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             })

        slice_node = Node(graph, 'strided_slice')
        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 3, 1, 2], inv=[0, 2, 3, 1]), 'begin_mask')
        self.assertTrue(np.array_equal(slice_node.begin_mask, np.array([1, 0, 1, 0])))

        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 3, 1, 2], inv=[0, 2, 3, 1]), 'end_mask')
        self.assertTrue(np.array_equal(slice_node.end_mask, np.array([0, 0, 1, 0])))

    def test_permute_begin_end_short(self):
        # Testing constant path case
        graph = build_graph(nodes_attributes,
                            [('data_1', 'strided_slice'),
                             ('begin', 'strided_slice'),
                             ('end', 'strided_slice'),
                             ('stride', 'strided_slice'),
                             ('strided_slice', 'data_2')],
                            {'data_1': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             'strided_slice': {'begin_mask': np.array([1, 0, 0]), 'end_mask': np.array([0, 1, 0]),
                                               'new_axis_mask': np.array([0, 0, 0]), 'shrink_axis_mask': [0, 0, 0],
                                               'ellipsis_mask': np.array([0, 0, 0])},
                             'data_2': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             })

        slice_node = Node(graph, 'strided_slice')
        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 3, 1, 2], inv=[0, 2, 3, 1]), 'begin_mask')
        self.assertTrue(np.array_equal(slice_node.begin_mask, np.array([1, 0, 0, 0])))

        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 3, 1, 2], inv=[0, 2, 3, 1]), 'end_mask')
        self.assertTrue(np.array_equal(slice_node.end_mask, np.array([0, 0, 1, 0])))

    def test_permute_begin_end_long(self):
        # Testing constant path case
        graph = build_graph(nodes_attributes,
                            [('data_1', 'strided_slice'),
                             ('begin', 'strided_slice'),
                             ('end', 'strided_slice'),
                             ('stride', 'strided_slice'),
                             ('strided_slice', 'data_2')],
                            {'data_1': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             'strided_slice': {'begin_mask': np.array([1, 0, 0, 1, 0]),
                                               'end_mask': np.array([0, 1, 0, 1, 1]),
                                               'new_axis_mask': np.array([0, 0, 0]),
                                               'shrink_axis_mask': [0, 0, 0],
                                               'ellipsis_mask': np.array([0, 0, 0])},
                             'data_2': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             })

        slice_node = Node(graph, 'strided_slice')
        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 3, 1, 2], inv=[0, 2, 3, 1]), 'begin_mask')
        self.assertTrue(np.array_equal(slice_node.begin_mask, np.array([1, 1, 0, 0, 0])))

        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 3, 1, 2], inv=[0, 2, 3, 1]), 'end_mask')
        self.assertTrue(np.array_equal(slice_node.end_mask, np.array([0, 1, 1, 0, 1])))

    def test_permute_begin_end_new(self):
        # Testing constant path case
        graph = build_graph(nodes_attributes,
                            [('data_1', 'strided_slice'),
                             ('begin', 'strided_slice'),
                             ('end', 'strided_slice'),
                             ('stride', 'strided_slice'),
                             ('strided_slice', 'data_2')],
                            {'data_1': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             'strided_slice': {'begin_mask': np.array([1, 0, 0, 1, 0]),
                                               'end_mask': np.array([0, 1, 0, 1, 1]),
                                               'new_axis_mask': np.array([1, 0, 0]),
                                               'shrink_axis_mask': [0, 0, 0],
                                               'ellipsis_mask': np.array([0, 0, 0])},
                             'data_2': {'shape': np.array([1, 1, 2, 3, 4]), 'value': None},
                             })

        slice_node = Node(graph, 'strided_slice')
        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 4, 1, 2, 3], inv=[0, 2, 3, 4, 1]), 'begin_mask')
        self.assertTrue(np.array_equal(slice_node.begin_mask, np.array([1, 0, 0, 0, 1])))

        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 4, 1, 2, 3], inv=[0, 2, 3, 4, 1]), 'end_mask')
        self.assertTrue(np.array_equal(slice_node.end_mask, np.array([0, 1, 1, 0, 1])))

    def test_permute_begin_end_new_short(self):
        # Testing constant path case
        graph = build_graph(nodes_attributes,
                            [('data_1', 'strided_slice'),
                             ('begin', 'strided_slice'),
                             ('end', 'strided_slice'),
                             ('stride', 'strided_slice'),
                             ('strided_slice', 'data_2')],
                            {'data_1': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             'strided_slice': {'begin_mask': np.array([1, 0, 0]), 'end_mask': np.array([0, 1, 0]),
                                               'new_axis_mask': np.array([1, 0, 0]), 'shrink_axis_mask': [0, 0, 0],
                                               'ellipsis_mask': np.array([0, 0, 0])},
                             'data_2': {'shape': np.array([1, 1, 2, 3, 4]), 'value': None},
                             })

        slice_node = Node(graph, 'strided_slice')
        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 4, 1, 2, 3], inv=[0, 2, 3, 4, 1]), 'begin_mask')
        self.assertTrue(np.array_equal(slice_node.begin_mask, np.array([1, 0, 0, 0, 0])))

        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 4, 1, 2, 3], inv=[0, 2, 3, 4, 1]), 'end_mask')
        self.assertTrue(np.array_equal(slice_node.end_mask, np.array([0, 0, 1, 0, 0])))

    def test_permute_begin_end_shrink(self):
        # Testing constant path case
        graph = build_graph(nodes_attributes,
                            [('data_1', 'strided_slice'),
                             ('begin', 'strided_slice'),
                             ('end', 'strided_slice'),
                             ('stride', 'strided_slice'),
                             ('strided_slice', 'data_2')],
                            {'data_1': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             'strided_slice': {'begin_mask': np.array([1, 0, 0, 1]), 'end_mask': np.array([0, 1, 0, 1]),
                                               'new_axis_mask': np.array([0, 0, 0]), 'shrink_axis_mask': [1, 0, 0],
                                               'ellipsis_mask': np.array([0, 0, 0])},
                             'data_2': {'shape': np.array([2, 3, 4]), 'value': None},
                             })

        slice_node = Node(graph, 'strided_slice')
        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 3, 1, 2], inv=[0, 2, 3, 1]), 'begin_mask')

        self.assertTrue(np.array_equal(slice_node.begin_mask, np.array([1, 1, 0, 0])))

        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 3, 1, 2], inv=[0, 2, 3, 1]), 'end_mask')
        self.assertTrue(np.array_equal(slice_node.end_mask, np.array([0, 1, 1, 0])))

    def test_permute_begin_end_shrink_short(self):
        # Testing constant path case
        graph = build_graph(nodes_attributes,
                            [('data_1', 'strided_slice'),
                             ('begin', 'strided_slice'),
                             ('end', 'strided_slice'),
                             ('stride', 'strided_slice'),
                             ('strided_slice', 'data_2')],
                            {'data_1': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             'strided_slice': {'begin_mask': np.array([1, 0, 0]), 'end_mask': np.array([0, 1, 0]),
                                               'new_axis_mask': np.array([0, 0, 0]), 'shrink_axis_mask': [1, 0, 0],
                                               'ellipsis_mask': np.array([0, 0, 0])},
                             'data_2': {'shape': np.array([2, 3, 4]), 'value': None},
                             })

        slice_node = Node(graph, 'strided_slice')
        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 3, 1, 2], inv=[0, 2, 3, 1]), 'begin_mask')
        self.assertTrue(np.array_equal(slice_node.begin_mask, np.array([1, 0, 0, 0])))

        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 3, 1, 2], inv=[0, 2, 3, 1]), 'end_mask')
        self.assertTrue(np.array_equal(slice_node.end_mask, np.array([0, 0, 1, 0])))

    def test_permute_begin_end_ellipsis(self):
        # Testing constant path case
        graph = build_graph(nodes_attributes,
                            [('input', 'data_1'),
                             ('data_1', 'strided_slice'),
                             ('begin', 'begin_data'),
                             ('begin_data', 'strided_slice'),
                             ('end', 'end_data'),
                             ('end_data', 'strided_slice'),
                             ('stride', 'stride_data'),
                             ('stride_data', 'strided_slice'),
                             ('strided_slice', 'data_2')],
                            {'data_1': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             'begin': {'value': [0, 1], 'shape': [2]},
                             'end': {'value': [1, 0], 'shape': [2]},
                             'stride': {'value': [1, 2], 'shape': [2]},
                             'strided_slice': {'begin_mask': np.array([0, 0]), 'end_mask': np.array([1, 0]),
                                               'new_axis_mask': np.array([0]), 'shrink_axis_mask': [0],
                                               'ellipsis_mask': np.array([1, 0])},
                             'data_2': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             })

        slice_node = Node(graph, 'strided_slice')
        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 3, 1, 2], inv=[0, 2, 3, 1]), 'begin_mask')
        self.assertTrue(np.array_equal(slice_node.begin_mask, np.array([0, 0, 0, 0])))

        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 3, 1, 2], inv=[0, 2, 3, 1]), 'end_mask')
        self.assertTrue(np.array_equal(slice_node.end_mask, np.array([1, 0, 0, 0])))

    def test_permute_begin_end_ellipsis_infer(self):
        # Testing constant path case
        graph = build_graph(nodes_attributes,
                            [('input', 'data_1'),
                             ('data_1', 'strided_slice', {'in': 0}),
                             ('begin', 'begin_data'),
                             ('begin_data', 'strided_slice', {'in': 1}),
                             ('end', 'end_data'),
                             ('end_data', 'strided_slice', {'in': 2}),
                             ('stride', 'stride_data'),
                             ('stride_data', 'strided_slice', {'in': 3}),
                             ('strided_slice', 'data_2')],
                            {'data_1': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             'begin': {'value': [0, 1], 'shape': [2]},
                             'end': {'value': [1, 0], 'shape': [2]},
                             'stride': {'value': [1, 2], 'shape': [2]},
                             'begin_data': {'value': [0, 1], 'shape': [2]},
                             'end_data': {'value': [1, 0], 'shape': [2]},
                             'stride_data': {'value': [1, 2], 'shape': [2]},
                             'strided_slice': {'begin_mask': np.array([0, 0]), 'end_mask': np.array([1, 0]),
                                               'new_axis_mask': np.array([0]), 'shrink_axis_mask': [0],
                                               'ellipsis_mask': np.array([1, 0])},
                             'data_2': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             })
        graph.graph['layout'] = "NHWC"

        slice_node = Node(graph, 'strided_slice')
        begin_node = Node(graph, 'begin')
        end_node = Node(graph, 'end')
        stride_node = Node(graph, 'stride')
        StridedSlice.infer(slice_node)
        self.assertTrue(np.array_equal(slice_node.begin_mask, np.array([0, 0, 0, 0])))
        self.assertTrue(np.array_equal(slice_node.end_mask, np.array([1, 0, 0, 0])))
        self.assertTrue(np.array_equal(slice_node.shrink_axis_mask, np.array([0, 0, 0, 0])))
        self.assertTrue(np.array_equal(slice_node.new_axis_mask, np.array([0, 0, 0, 0])))
        self.assertTrue(np.array_equal(begin_node.value, np.array([0, 1, 0, 0])))
        self.assertTrue(np.array_equal(end_node.value, np.array([1, 0, 0, 0])))
        self.assertTrue(np.array_equal(stride_node.value, np.array([1, 2, 1, 1])))

    def test_permute_begin_end_ellipsis_new(self):
        # Testing constant path case
        graph = build_graph(nodes_attributes,
                            [('input', 'data_1'),
                             ('data_1', 'strided_slice', {'in': 0}),
                             ('begin', 'begin_data'),
                             ('begin_data', 'strided_slice', {'in': 1}),
                             ('end', 'end_data'),
                             ('end_data', 'strided_slice', {'in': 2}),
                             ('stride', 'stride_data'),
                             ('stride_data', 'strided_slice', {'in': 3}),
                             ('strided_slice', 'data_2')],
                            {'data_1': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             'begin': {'value': [0, 1, 0], 'shape': [3]},
                             'begin_data': {'value': [0, 1, 0], 'shape': [3]},
                             'end': {'value': [1, 0, 1], 'shape': [3]},
                             'end_data': {'value': [1, 0, 1], 'shape': [3]},
                             'stride': {'value': [1, 2, 3], 'shape': [3]},
                             'stride_data': {'value': [1, 2, 3], 'shape': [3]},
                             'strided_slice': {'begin_mask': np.array([1, 2, 3]), 'end_mask': np.array([1, 2, 3]),
                                               'new_axis_mask': np.array([1, 0, 0]), 'shrink_axis_mask': [0],
                                               'ellipsis_mask': np.array([0, 1, 0])},
                             'data_2': {'shape': np.array([1, 1, 2, 3, 4]), 'value': None},
                             })

        slice_node = Node(graph, 'strided_slice')
        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 4, 1, 2, 3], inv=[0, 2, 3, 4, 1]), 'begin_mask')
        self.assertTrue(np.array_equal(slice_node.begin_mask, np.array([1, 3, 2, 0, 0])))

        permute_masks(slice_node, PermuteAttrs.Permutation(perm=[0, 4, 1, 2, 3], inv=[0, 2, 3, 4, 1]), 'end_mask')
        self.assertTrue(np.array_equal(slice_node.end_mask, np.array([1, 3, 2, 0, 0])))

    def test_permute_begin_end_ellipsis_new_inputs(self):
        # Testing constant path case
        graph = build_graph(nodes_attributes,
                            [('input', 'data_1'),
                             ('data_1', 'strided_slice', {'in': 0}),
                             ('begin', 'begin_data'),
                             ('begin_data', 'strided_slice', {'in': 1}),
                             ('end', 'end_data'),
                             ('end_data', 'strided_slice', {'in': 2}),
                             ('stride', 'stride_data'),
                             ('stride_data', 'strided_slice', {'in': 3}),
                             ('strided_slice', 'data_2')],
                            {'data_1': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             'strided_slice': {'begin_mask': np.array([0, 0, 0]), 'end_mask': np.array([1, 0, 0]),
                                               'new_axis_mask': np.array([1, 0, 0]), 'shrink_axis_mask': [0],
                                               'ellipsis_mask': np.array([0, 1, 0])},
                             'begin': {'value': np.array([0, 1, 2])},
                             'end': {'value': np.array([1, 2, 3])},
                             'stride': {'value': np.array([1, 1, 1])},
                             'begin_data': {'value': np.array([0, 1, 2])},
                             'end_data': {'value': np.array([1, 2, 3])},
                             'stride_data': {'value': np.array([1, 1, 1])},
                             'data_2': {'shape': np.array([1, 1, 2, 3, 4]), 'value': None},
                             })

        slice_node = Node(graph, 'strided_slice')
        slice_node.in_node(1).value = permute_array_with_ellipsis(slice_node,
                                                                  slice_node.in_node(1).value, 0)
        self.assertTrue(np.array_equal(slice_node.in_node(1).value, np.array([0, 2, 1, 0, 0])))

        slice_node.in_node(2).value = permute_array_with_ellipsis(slice_node,
                                                                  slice_node.in_node(2).value, 0)
        self.assertTrue(np.array_equal(slice_node.in_node(2).value, np.array([1, 3, 2, 0, 0])))

    def test_extend_mask(self):
        ellipsis_mask = int64_array([1, 0])
        shrink_mask = int64_array([0, 0])
        length_shape = 4
        mask = int64_array([0, 1])
        ins_value = 0
        mask = extend_mask_according_ellipsis(ellipsis_mask, shrink_mask, length_shape, list(mask), ins_value)
        self.assertEquals(mask, [0, 0, 0, 1])

    def test_extend_mask_shrinked(self):
        ellipsis_mask = int64_array([1, 0])
        shrink_mask = int64_array([0, 1])
        length_shape = 4
        mask = int64_array([0, 1])
        ins_value = 2
        mask = extend_mask_according_ellipsis(ellipsis_mask, shrink_mask, length_shape, list(mask), ins_value)
        self.assertEquals(mask, [0, 2, 2, 2, 1])

    def test_extend_mask_shrinked_shrink_mask(self):
        ellipsis_mask = int64_array([0, 1, 0])
        shrink_mask = int64_array([0, 0, 1])
        length_shape = 4
        ins_value = 2
        shrink_mask = extend_mask_according_ellipsis(ellipsis_mask, shrink_mask, length_shape, list(shrink_mask),
                                                     ins_value)
        self.assertEquals(shrink_mask, [0, 0, 2, 2, 1])

    def test_non_const_infer(self):
        # Testing constant path case
        graph = build_graph(nodes_attributes,
                            [('input', 'data_1'),
                             ('data_1', 'strided_slice', {'in': 0}),
                             ('data_1', 'strided_slice', {'in': 1}),
                             ('end', 'end_data'),
                             ('end_data', 'strided_slice', {'in': 2}),
                             ('stride', 'stride_data'),
                             ('stride_data', 'strided_slice', {'in': 3}),
                             ('strided_slice', 'data_2')],
                            {'data_1': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             'end': {'value': [1, 0], 'shape': [2]},
                             'stride': {'value': [1, 2], 'shape': [2]},
                             'strided_slice': {'begin_mask': np.array([0, 0]), 'end_mask': np.array([1, 0]),
                                               'new_axis_mask': np.array([0]), 'shrink_axis_mask': [0],
                                               'ellipsis_mask': np.array([1, 0])},
                             'data_2': {'shape': np.array([1, 2, 3, 4]), 'value': None},
                             })
        graph.graph['layout'] = "NHWC"

        slice_node = Node(graph, 'strided_slice')
        with self.assertRaises(Error) as error:
            StridedSlice.infer(slice_node)
        self.assertTrue('Strided slice layer supports only constant begin and end inputs' in str(error.exception))
