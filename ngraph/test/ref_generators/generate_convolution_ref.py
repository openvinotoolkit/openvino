#!/usr/bin/env python
# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import sys
import numpy as np
import math
import random
from operator import mul

# Generates an array of random floating point literals of the given length, from a fixed seed.


def random_array_float_literals(length, seed=8086):
    literals = []

    random.seed(seed)

    for i in range(0, length):
        # generate numbers that can be exactly represented in binary
        sig_bits = 6
        range_bits = 2
        literal_n = np.float32(random.randint(-pow(2, sig_bits-1),
                                              pow(2, sig_bits-1))) / pow(2.0, sig_bits - range_bits)
        literals.append(str(literal_n))

    return literals

# Elementwise addition on tuples.


def tuple_plus(t1, t2):
    assert(len(t1) == len(t2))

    res = ()

    for (x, y) in zip(list(t1), list(t2)):
        res = res + (x+y,)

    return res

# Elementwise multiplication on tuples.


def tuple_times(t1, t2):
    assert(len(t1) == len(t2))

    res = ()

    for (x, y) in zip(list(t1), list(t2)):
        res = res + (x*y,)

    return res

#
# Convolution reference
#
#    Arguments:
#    data_batch       : [N ][Ci][D1]...[Dn], n > 0
#    filter           : [Co][Ci][W1]...[Wn]
#    move_strides     = (s1,...,sn)
#    filter_dilation  = (l1,...,ln)
#    below_pads       = (p1,...,pn)
#    above_pads       = (q1,...,qn)
#    data_dilation    = (t1,...,tn)
#
#    Returns:
#    output_batch     : [N ][Co][D'1]...[D'n]
#
# Where the D's are computed according to TensorFlow-style "valid" convolution rules, but *after* padding.
# See https://www.tensorflow.org/api_docs/python/tf/nn/convolution.
#


def convolution_ref(data_batch, filter, move_strides, filter_dilation, below_pads, above_pads, data_dilation):
    assert(len(data_batch.shape) == len(filter.shape))
    assert(len(data_batch.shape) > 2)
    assert(len(data_batch.shape) <= 6)
    assert(data_batch.shape[1] == filter.shape[1])
    assert(len(move_strides) == len(data_batch.shape) - 2)
    assert(len(filter_dilation) == len(data_batch.shape) - 2)
    assert(len(data_dilation) == len(data_batch.shape) - 2)

    # dilate the input batch
    new_item_shape = (np.array(data_batch.shape[2:]) - 1) * data_dilation + 1
    new_data_batch_shape = list(
        np.array(data_batch.shape[:2])) + list(new_item_shape)
    new_data_batch = np.zeros(new_data_batch_shape)

    for n in range(0, new_data_batch_shape[0]):
        for c in range(0, new_data_batch_shape[1]):
            if new_data_batch.ndim == 3:
                new_data_batch[n, c, 0::data_dilation[0]] = data_batch[n][c]
            elif new_data_batch.ndim == 4:
                new_data_batch[n, c, 0::data_dilation[0],
                               0::data_dilation[1]] = data_batch[n][c]
            elif new_data_batch.ndim == 5:
                new_data_batch[n, c, 0::data_dilation[0],
                               0::data_dilation[1], 0::data_dilation[2]] = data_batch[n][c]
            elif new_data_batch.ndim == 6:
                new_data_batch[n, c, 0::data_dilation[0], 0::data_dilation[1],
                               0::data_dilation[2], 0::data_dilation[3]] = data_batch[n][c]
            else:
                assert(False)

    data_batch = new_data_batch

    # Pad the input batch wherever the pads are positive.
    # Have to add values for the spatial and channel dims.
    below_pads_pos = (0, 0) + tuple(np.clip(below_pads, 0, None))
    # Have to add values for the spatial and channel dims.
    above_pads_pos = (0, 0) + tuple(np.clip(above_pads, 0, None))
    data_batch = np.pad(data_batch, list(
        zip(below_pads_pos, above_pads_pos)), mode='constant', constant_values=0)

    # Slice the input batch wherever the pads are negative.
    slice_bottoms = (0, 0) + tuple(-np.clip(below_pads, None, 0))
    slice_tops = (0, 0) + tuple(np.clip(above_pads, None, 0))
    slices = list(map(lambda p: slice(
        p[0], p[1] if p[1] < 0 else None), zip(slice_bottoms, slice_tops)))
    data_batch = data_batch[tuple(slices)]

    item_count = data_batch.shape[0]               # N
    ci_count = data_batch.shape[1]                 # Ci
    co_count = filter.shape[0]                     # Co
    input_item_shape = list(data_batch.shape[2:])  # D1, ..., Dn
    window_virtual_shape = list(filter.shape[2:])  # W1, ..., Wn

    # This is not used in computation but we will calculate it for a check to make sure the window fits.
    window_physical_shape = []
    for (d_in, d_virt, dil) in zip(input_item_shape, window_virtual_shape, filter_dilation):
        d_phys = (d_virt - 1) * dil + 1
        assert(d_phys <= d_in)
        window_physical_shape.append(d_phys)

    output_item_shape = []  # D'1,...,D'n
    for (d_in, d_win, dil, mov) in zip(input_item_shape, window_virtual_shape, filter_dilation, move_strides):
        # Formula is taken from TF's definition for VALID convolution.
        d_out = int(
            math.ceil((float(d_in) - (float(d_win) - 1.0) * float(dil))/float(mov)))
        assert(d_out > 0)
        output_item_shape.append(d_out)

    output_shape = [item_count, co_count]+output_item_shape  # N,Co,D'1,...,D'n
    output_batch = np.zeros(output_shape)

    # Walk over the output batch space.
    output_it = np.nditer(output_batch, flags=['multi_index'])
    while not output_it.finished:
        # Break up the output coordinate to figure out where we are in terms of batch index, output channel, and spatial position.
        output_index = output_it.multi_index
        item, co, output_pos = output_index[0], output_index[1], output_index[2:]

        # Walk over the filter for the current output channel.
        filter_it = np.nditer(filter[co], flags=['multi_index'])
        while not filter_it.finished:
            # Break up the filter coordinate to figure out where we are in terms of input channel and filter shape position.
            filter_index = filter_it.multi_index
            ci, filter_pos = filter_index[0], filter_index[1:]

            # Build up the coordinate within the space N,Ci,D1,...,Dn that we need to read from in the input batch.
            input_index = (item, ci) + (tuple_plus(tuple_times(output_pos,
                                                               move_strides), tuple_times(filter_pos, filter_dilation)))

            # Add to the sum-of-products.
            output_batch[output_index] = output_batch[output_index] + \
                filter[(co,) + filter_index] * data_batch[input_index]

            filter_it.iternext()

        output_it.iternext()

    return output_batch


def shape_str(shape):
    result = ''
    first = True
    for d in shape:
        if first:
            result = ('%d' % d)
            first = False
        else:
            result = result + (',%d' % d)
    return result


def scalar_str(x):
    result = ('%.1000g' % x)
    # This next part is a bit stupid.
    if "." not in result and "e" not in result:
        result = result + ".0f"
    else:
        result = "%.8ff" % float(result)
    return result


def data_str(data):
    result = ''
    first = True
    for x in np.nditer(data):
        if first:
            result = scalar_str(x)
            first = False
        else:
            result = result + ',' + scalar_str(x)
    return result


def shape_size(shape):
    result = 1
    for l in shape:
        result = result * l
    return result


def emit_test(t, f):
    test_name, input_batch_shape, filters_shape, move_strides, filter_dilation, below_pads, above_pads, data_dilation, bprop = t

    input_batch_literals = random_array_float_literals(
        shape_size(input_batch_shape))
    filters_literals = random_array_float_literals(shape_size(filters_shape))
    input_batch_array = np.array(
        list(map(lambda s: np.float32(s), input_batch_literals)))
    input_batch_array.shape = input_batch_shape
    filters_array = np.array(
        list(map(lambda s: np.float32(s), filters_literals)))
    filters_array.shape = filters_shape

    print("Generating convolution test '%s'..." % test_name)

    output_batch_data = convolution_ref(
        input_batch_array, filters_array, move_strides, filter_dilation, below_pads, above_pads, data_dilation)

    template = '''
// !!!!!!!!!!!!!! THIS FILE IS AUTOGENERATED OUTSIDE OF THE BUILD PROCESS !!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT EDIT THIS FILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
// DO NOT EDIT THIS FILE. If you want to add new tests, you should edit
//  test/ref_generators/generate_convolution_ref.py and regenerate this file.
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT EDIT THIS FILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!! THIS FILE IS AUTOGENERATED OUTSIDE OF THE BUILD PROCESS !!!!!!!!!!!!!!
NGRAPH_TEST (${BACKEND_NAME}, %s)
{
    Shape shape_a{%s};
    Shape shape_b{%s};
    Shape shape_r{%s};
    auto make_graph = [shape_a, shape_b] {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        auto B = make_shared<op::Parameter>(element::f32, shape_b);
        return make_shared<Function>(make_shared<op::Convolution>(A, B,
                                                                  Strides{%s},        // move_strides
                                                                  Strides{%s},        // filter_dilation
                                                                  CoordinateDiff{%s}, // below_pads
                                                                  CoordinateDiff{%s}, // above_pads
                                                                  Strides{%s}),       // data_dilation
                                     ParameterVector{A, B});
    };

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto function = make_graph();

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{%s});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{%s});
    auto result = backend->create_tensor(element::f32, shape_r);

    vector<float> expected_result{%s};

    auto handle = backend->compile(function);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(vector<float>{expected_result}, read_vector<float>(result), tolerance));
    // only test backprop for certain cases as it takes significant compute resources
    %sEXPECT_TRUE(autodiff_numeric_compare<float>(backend.get(), make_graph, {a, b}, .01f, .01f));
}
'''
    f.write(template % (test_name,
                        shape_str(input_batch_shape),
                        shape_str(filters_shape),
                        shape_str(output_batch_data.shape),
                        shape_str(move_strides),
                        shape_str(filter_dilation),
                        shape_str(below_pads),
                        shape_str(above_pads),
                        shape_str(data_dilation),
                        ",".join(map(lambda s: "%.8ff" %
                                     float(s), input_batch_literals)),
                        ",".join(map(lambda s: "%.8ff" %
                                     float(s), filters_literals)),
                        data_str(output_batch_data),
                        bprop))


#                                                                              filter                                      data
#         test name                                skip list   i             batch shape   filts shape   stride    dilation  below-pads  above-pads  dilation   bprop?
tests = [
    ("convolution_2d_1item",                  (1, 1, 3, 5),    (2, 1, 2, 2),
     (1, 1),    (1, 1),    (0, 0),      (0, 0),      (1, 1),     ""),
    ("convolution_2d_1item_padded_1_1x1_1",   (1, 1, 3, 5),    (2, 1, 2, 2),
     (1, 1),    (1, 1),    (1, 1),      (1, 1),      (1, 1),     ""),
    ("convolution_2d_1item_padded_2_3x4_5",   (1, 1, 3, 5),    (2, 1, 2, 2),
     (1, 1),    (1, 1),    (2, 3),      (4, 5),      (1, 1),     ""),
    ("convolution_2d_2items",                 (2, 1, 3, 5),    (2, 1, 2, 2),
     (1, 1),    (1, 1),    (0, 0),      (0, 0),      (1, 1),     ""),
    ("convolution_2d_2items_strided",         (2, 1, 3, 5),    (2, 1, 2, 2),
     (2, 2),    (1, 1),    (0, 0),      (0, 0),      (1, 1),     ""),
    ("convolution_2d_2items_strided_padded",  (2, 1, 3, 5),    (2, 1, 2, 2),
     (2, 2),    (1, 1),    (4, 2),      (5, 7),      (1, 1),     ""),
    ("convolution_2d_2items_strided_padded_same", (2, 1, 3, 5), (2, 1, 2, 2),
     (2, 2),    (1, 1),    (2, 2),      (2, 2),      (1, 1),     ""),
    ("convolution_2d_2items_dilated",         (2, 1, 3, 5),    (2, 1, 2, 2),
     (1, 1),    (2, 2),    (0, 0),      (0, 0),      (1, 1),     ""),
    ("convolution_2d_2items_dilated_padded",  (2, 1, 3, 5),    (2, 1, 2, 2),
     (1, 1),    (2, 2),    (4, 2),      (5, 7),      (1, 1),     ""),
    ("convolution_3d_2items",                 (2, 1, 3, 5, 8),  (2, 1, 2, 2, 3),
     (1, 1, 1),  (1, 1, 1),  (0, 0, 0),    (0, 0, 0),    (1, 1, 1),   ""),
    ("convolution_4d_2items",                 (2, 1, 3, 5, 8, 7), (2, 1, 2, 2, 3, 1),
     (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0),  (0, 0, 0, 0),  (1, 1, 1, 1), "// "),
    ("convolution_4d_4items",                 (4, 3, 3, 5, 8, 7), (4, 3, 2, 2, 3, 1),
     (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0),  (0, 0, 0, 0),  (1, 1, 1, 1), "// "),
    ("convolution_4d_4items_padded_neg",      (4, 3, 3, 5, 8, 7), (4, 3, 2, 2, 3, 1),
     (1, 1, 1, 1), (1, 1, 1, 1), (-1, 2, -3, 2), (1, 0, 0, -3), (1, 1, 1, 1), "// "),
    ("convolution_4d_4items_strided",         (4, 3, 3, 5, 8, 7), (4, 3, 2, 2, 3, 1),
     (2, 1, 3, 2), (1, 1, 1, 1), (0, 0, 0, 0),  (0, 0, 0, 0),  (1, 1, 1, 1), "// "),
    ("convolution_4d_4items_dilated",         (4, 3, 3, 5, 8, 7), (4, 3, 2, 2, 3, 1),
     (1, 1, 1, 1), (2, 1, 3, 2), (0, 0, 0, 0),  (0, 0, 0, 0),  (1, 1, 1, 1), "// "),
    ("convolution_4d_4items_strided_dilated", (4, 3, 8, 8, 8, 8), (4, 3, 2, 2, 3, 1),
     (3, 2, 2, 3), (2, 1, 3, 2), (0, 0, 0, 0),  (0, 0, 0, 0),  (1, 1, 1, 1), "// "),
    ("convolution_4d_4items_strided_dilated_padded",
     (4, 3, 8, 8, 8, 8), (4, 3, 2, 2, 3, 1), (3, 2, 2, 3), (2, 1, 3, 2), (2, 4, 6, 8),  (1, 3, 5, 7),  (1, 1, 1, 1), "// "),
    ("convolution_4d_4items_strided_dilated_padded_neg",
     (4, 3, 8, 8, 8, 8), (4, 3, 2, 2, 3, 1), (3, 2, 2, 3), (2, 1, 3, 2), (-2, 4, 0, 5), (1, 3, -1, -4), (1, 1, 1, 1), "// "),
    ("convolution_4d_4items_strided_dilated_padded_same",
     (4, 3, 8, 8, 8, 8), (4, 3, 2, 2, 3, 1), (3, 2, 2, 3), (2, 1, 3, 2), (3, 3, 3, 3),  (3, 3, 3, 3),  (1, 1, 1, 1), "// "),
    ("convolution_2d_1item_1o1i_data_dilated", (1, 1, 3, 5),    (1, 1, 2, 2),
     (1, 1),    (1, 1),    (0, 0),      (0, 0),      (2, 2),     ""),
    ("convolution_2d_1item_2o1i_data_dilated", (1, 1, 3, 5),    (2, 1, 2, 2),
     (1, 1),    (1, 1),    (0, 0),      (0, 0),      (2, 2),     ""),
    ("convolution_2d_1item_2o2i_data_dilated", (1, 2, 3, 5),    (2, 2, 2, 2),
     (1, 1),    (1, 1),    (0, 0),      (0, 0),      (2, 2),     ""),
    ("convolution_2d_1item_5o3i_data_dilated", (1, 3, 3, 5),    (5, 3, 2, 2),
     (1, 1),    (1, 1),    (0, 0),      (0, 0),      (2, 2),     ""),
    ("convolution_2d_2item_5o3i_data_dilated", (2, 3, 3, 5),    (5, 3, 2, 2),
     (1, 1),    (1, 1),    (0, 0),      (0, 0),      (2, 2),     ""),
    ("convolution_2d_8item_large_5o3i_data_dilated",
     (8, 3, 16, 16),  (5, 3, 2, 2),    (1, 1),    (1, 1),    (0, 0),      (0, 0),      (2, 2),     "// "),
    ("convolution_2d_8item_large_5o3i_uneven_filter_data_dilated",
     (8, 3, 16, 16),  (5, 3, 2, 3),    (1, 1),    (1, 1),    (0, 0),      (0, 0),      (2, 2),     "// "),
    ("convolution_2d_8item_large_5o3i_uneven_filter_uneven_data_dilation_data_dilated",
     (8, 3, 16, 16),  (5, 3, 2, 3),    (1, 1),    (1, 1),    (0, 0),      (0, 0),      (2, 3),     "// "),
    ("convolution_3d_2item_large_5o3i_uneven_filter_uneven_data_dilation_data_dilated",
     (2, 3, 8, 8, 8),  (5, 3, 2, 3, 4),  (1, 1, 1),  (1, 1, 1),  (0, 0, 0),    (0, 0, 0),    (2, 3, 2),   "// "),
    ("convolution_3d_1item_large_5o3i_padded_uneven_filter_uneven_data_dilation_data_dilated",
     (1, 3, 8, 8, 8),  (5, 3, 2, 3, 4),  (1, 1, 1),  (1, 1, 1),  (2, 1, 2),    (1, 2, 3),    (2, 3, 2),   "// "),
    ("convolution_3d_2item_large_5o3i_padded_strided_uneven_filter_uneven_data_dilation_data_dilated",
     (2, 3, 8, 8, 8),  (5, 3, 2, 3, 4),  (2, 3, 2),  (1, 1, 1),  (2, 1, 2),    (1, 2, 3),    (2, 3, 2),   "// "),
    ("convolution_3d_2item_large_5o3i_padded_strided_uneven_filter_uneven_data_dilation_filter_dilated_data_dilated",
     (2, 3, 8, 8, 8),  (5, 3, 2, 3, 4),  (2, 3, 2),  (3, 2, 2),  (2, 1, 2),    (1, 2, 3),    (2, 3, 2),   "// "),
]

def main():
    assert(len(sys.argv) > 1)

    f = open(sys.argv[1], 'w')
    f.write('''//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

// !!!!!!!!!!!!!! THIS FILE IS AUTOGENERATED OUTSIDE OF THE BUILD PROCESS !!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT EDIT THIS FILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
// It takes quite a while to compute the results.
//
// DO NOT EDIT THIS FILE. If you want to add new tests, you should edit
//  test/ref_generators/generate_convolution_ref.py and regenerate this file.
//
// To regenerate:
//
//   $ cd <ngraph source dir>/test
//   $ ./update_convolution_reference.sh
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT EDIT THIS FILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!! THIS FILE IS AUTOGENERATED OUTSIDE OF THE BUILD PROCESS !!!!!!!!!!!!!!
//
// clang-format off

#include <cmath>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "util/test_tools.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/all_close_f.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

// for float this will be 18 bits matching
// for bfloat this will be 6 bits matching
constexpr int three_quarters_of_available_bits = (MAX_FLOAT_BITS * 3) / 4;
constexpr int tolerance = FLOAT_MANTISSA_BITS - three_quarters_of_available_bits;

''')

    for t in tests:
        emit_test(t, f)

    f.write('''
// clang-format on
''')

    f.close()


if __name__ == "__main__":
    main()
