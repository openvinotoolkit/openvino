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

#
# Test case generator for DynReplaceSlice op.
#
# TODO(amprocte): de-duplicate lots of code in generate_dyn_slice_ref.py.
#

import sys
import numpy as np

def make_iterable(x):
    try:
        _ = iter(x)
    except TypeError as _:
        return [x]
    return x

def print_lb_values(slices):
    slices = make_iterable(slices)

    strs = []

    for sl in slices:
        try:
            x = int(sl)
            strs.append(str(x))
        except TypeError as _:
            if isinstance(sl, slice) and sl.start is not None:
                strs.append(str(sl.start))
            else:
                strs.append('0')
    return ','.join(strs)

def print_ub_values(slices):
    slices = make_iterable(slices)

    strs = []

    for sl in slices:
        if isinstance(sl, slice) and sl.stop is not None:
            strs.append(str(sl.stop))
        else:
            strs.append('0')
    return ','.join(strs)

def print_stride_values(slices):
    slices = make_iterable(slices)

    strs = []

    for sl in slices:
        if isinstance(sl, slice) and sl.step is not None:
            strs.append(str(sl.step))
        else:
            strs.append('1')
    return ','.join(strs)

def print_lb_mask_axes(slices):
    slices = make_iterable(slices)

    mask_strs = []
    i = 0

    for sl in slices:
        if isinstance(sl, slice) and sl.start is None:
            mask_strs.append(str(i))
        i += 1
    return ','.join(mask_strs)

def print_ub_mask_axes(slices):
    slices = make_iterable(slices)

    mask_strs = []
    i = 0

    for sl in slices:
        if isinstance(sl, slice) and sl.stop is None:
            mask_strs.append(str(i))
        i += 1
    return ','.join(mask_strs)

def print_new_mask_axes(slices):
    slices = make_iterable(slices)

    mask_strs = []
    i = 0

    for sl in slices:
        if sl is None:
            mask_strs.append(str(i))
        i += 1
    return ','.join(mask_strs)

def print_shrink_mask_axes(slices):
    slices = make_iterable(slices)

    mask_strs = []
    i = 0

    for sl in slices:
        try:
            _ = int(sl)
            mask_strs.append(str(i))
        except TypeError as _:
            pass
        i += 1
    return ','.join(mask_strs)

def print_ellipsis_mask_axes(slices):
    slices = make_iterable(slices)

    mask_strs = []
    i = 0

    for sl in slices:
        if sl is Ellipsis:
            mask_strs.append(str(i))
        i += 1
    return ','.join(mask_strs)

def np_dt_to_c(dtype):
    if dtype=='int8':
        return 'int8_t'
    elif dtype=='uint8':
        return 'uint8_t'
    elif dtype=='int16':
        return 'int16_t'
    elif dtype=='uint16':
        return 'uint16_t'
    elif dtype=='int32':
        return 'int32_t'
    elif dtype=='uint32':
        return 'uint32_t'
    elif dtype=='int64':
        return 'int64_t'
    elif dtype=='uint64':
        return 'uint64_t'
    elif dtype=='float16':
        return 'float16'
    elif dtype=='float32':
        return 'float'
    elif dtype=='float64':
        return 'double'
    elif dtype=='bool':
        return 'char'
    else:
        raise ValueError('Unsupported numpy data type: %s' % dtype)

def np_dt_to_ng(dtype):
    if dtype=='int8':
        return 'element::i8'
    elif dtype=='uint8':
        return 'element::u8'
    elif dtype=='int16':
        return 'element::i16'
    elif dtype=='uint16':
        return 'element::u16'
    elif dtype=='int32':
        return 'element::i32'
    elif dtype=='uint32':
        return 'element::u32'
    elif dtype=='int64':
        return 'element::i64'
    elif dtype=='uint64':
        return 'element::u64'
    elif dtype=='float16':
        return 'element::f16'
    elif dtype=='float32':
        return 'element::f32'
    elif dtype=='float64':
        return 'element::f64'
    elif dtype=='bool':
        return 'element::boolean'
    else:
        raise ValueError('Unsupported numpy data type: %s' % dtype)

def print_values(values):
    values = make_iterable(values)
    strs = []

    for v in values:
        strs.append(str(v))

    return ','.join(strs)

def print_shape(dims):
    dims = make_iterable(dims)
    strs = []

    for d in dims:
        strs.append(str(d))

    return 'Shape{' + ','.join(strs) + '}'

def print_slice(sl):
    if sl is None:
        return 'newaxis'
    elif sl is Ellipsis:
        return "..."
    elif isinstance(sl, slice):
        s = ''
        if sl.start is not None:
            s += str(sl.start)
        s += ':'
        if sl.stop is not None:
            s += str(sl.stop)
        if sl.step is not None:
            s += ':'
            s += str(sl.step)
        return s
    else:
        return str(sl)

def print_slices(slices):
    slices = make_iterable(slices)
    strs = []

    for sl in slices:
        strs.append(print_slice(sl))

    return '[' + ','.join(strs) + ']'

#
# Class to intercept __setitem__ operations and write an nGraph C++ test case.
# The generated test case will ensure that the output is identical to what
# would be produced by numpy. Specifically, the numpy (and equivalent C++)
# it will generate a "linspaced" array of the appropriate shape and dtype, and
# attempt to overwrite it with the "value" argument of __setitem__. If the
# value is None, it will auto-generate a replacement value of the appropriate
# shape.
#
# We will attempt to catch any exceptions raised by numpy's __setitem__, and
# generate a test that checks for erroring behavio.
#
# Example usage:
#
#    w = ReplaceSliceTestWriter(stream=sys.stdout)
#
#    # behave as if writing into a 4x5x6 input array of data type int32
#    w.set_shape(4,5,6)
#    w.set_dtype('int32')
#
#    # generate test cases for various behaviors, writing C++ code to sys.stdout
#    w[0,:,:]       = np.ones(shape=(5,6), dtype='int32')
#    w[0,:,:]       = None   # test will auto-generate something of shape (5,6)
#    w[...,-1:-3,:] = np.zeros(shape=(4,0,6), dtype='int32')
#
#    # generate test cases for some erroring behaviors, writing C++ code to
#    # sys.stdout
#    w[1,2,3,4] = 0                # too many indices
#    w[7] = np.ones(shape=(5,6))   # index out of bounds
#    w[1,1] = [2]                  # shape mismatch between slice and
#                                  # replacement (NOTE: this example is
#                                  # actually legal in np because it would
#                                  # auto-broadcast, but not in nGraph.)
#
class ReplaceSliceTestWriter:
    def __init__(self, shape=(), dtype='int32', stream=sys.stdout):
        self._shape = shape
        self._dtype = dtype
        self._stream = stream
        self._test_counter = 0

    def __setitem__(self, slices, value):
        self.write_test(slices, value)

    def write_test(self, slices, value, value_shape=None):
        # Generate some linspaced input data.
        data_in = np.linspace(0,np.prod(self._shape)-1,np.prod(self._shape),dtype=self._dtype).reshape(self._shape)

        failure_reasons = []

        if value_shape is None:
            try:
                slice_shape = data_in.__getitem__(slices).shape
            except Exception:
                failure_reasons.append('numpy getitem failed')
                slice_shape = ()

        if value_shape is None:
            value_shape = slice_shape

        # If `value` is None, we'll auto-generate some data. This will only
        # work if value_shape is specified, OR if the slices are legal for the
        # input.
        #
        # Generated value is linspaced, starting where data_in left off.
        if value is None:
            value = np.linspace(np.prod(self._shape), np.prod(self._shape) + np.prod(value_shape) - 1, np.prod(value_shape), dtype=self._dtype).reshape(value_shape)
        else:
            value = np.array(value)

        # numpy allows autobroadcast of the replacement to match the slice
        # shape, but we don't, so we would expect failure in that case
        if value.shape != slice_shape:
            failure_reasons.append('slice shape and replacement shape do not match')

        self._stream.write('\n')
        self._stream.write('                                       // test %d\n' % self._test_counter)
        self._stream.write('                                       // slices are: %s\n' % print_slices(slices))
        self._stream.write('                                       // dtype is: %s\n' % self._dtype)
        self._stream.write('                                       // input shape is: %s\n' % print_shape(self._shape))
        self._stream.write('                                       // slice shape is: %s\n' % print_shape(slice_shape))
        self._stream.write('                                       // replacement shape is: %s\n' % print_shape(value.shape))

        # If numpy fails for any reason, we expect failure.
        try:
            data_out = data_in
            data_out.__setitem__(slices, value)
        except Exception:
            failure_reasons.append('numpy setitem failed')

        # numpy allows implicit data type conversion, but we don't, so we
        # expect failure if dtypes do not match.
        if value.dtype != self._dtype:
            failure_reasons.append('dtype mismatch')

        is_failed = (failure_reasons != [])

        if is_failed:
            result_values = np.array([], dtype=self._dtype)
        else:
            result_values = data_out

        if is_failed:
            self._stream.write('                                       // failure is expected (%s)\n' % ','.join(failure_reasons))
        else:
            self._stream.write('                                       // expected output shape is %s\n' % print_shape(data_in.shape))

        self._stream.write('                                       make_shared<DynReplaceSliceTestParams<%s,%s>>(\n'
                           '                                           %s,\n'
                           '                                           %s,\n'
                           '                                           %s,\n'
                           '                                           %s,\n'
                           '                                           %s,\n'
                           '                                           std::vector<int64_t>{%s},\n'
                           '                                           std::vector<int64_t>{%s},\n'
                           '                                           std::vector<int64_t>{%s},\n'
                           '                                           AxisSet{%s},\n'
                           '                                           AxisSet{%s},\n'
                           '                                           AxisSet{%s},\n'
                           '                                           AxisSet{%s},\n'
                           '                                           AxisSet{%s},\n'
                           '                                           std::vector<%s>{%s},\n'
                           '                                           std::vector<%s>{%s}\n'
                           '                                       ),\n'
                                % (np_dt_to_c(self._dtype), np_dt_to_c(value.dtype),

                                    'false' if is_failed else 'true',

                                    np_dt_to_ng(self._dtype),
                                    np_dt_to_ng(value.dtype),
                                    print_shape(data_in.shape),
                                    print_shape(value.shape),

                                    print_lb_values(slices),
                                    print_ub_values(slices),
                                    print_stride_values(slices),

                                    print_lb_mask_axes(slices),
                                    print_ub_mask_axes(slices),
                                    print_new_mask_axes(slices),
                                    print_shrink_mask_axes(slices),
                                    print_ellipsis_mask_axes(slices),

                                    np_dt_to_c(self._dtype), print_values(result_values.reshape(-1)),
                                    np_dt_to_c(value.dtype), print_values(value.reshape(-1))))

        self._test_counter += 1

    def set_shape(self,shape):
        self._shape = shape

    def set_dtype(self,dtype):
        self._dtype = dtype

def write_header(f):
    f.write('''\
//*****************************************************************************
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
// DO NOT EDIT THIS FILE. If you want to add new tests, you should edit
//  test/ref_generators/generate_dyn_replace_slice_ref.py and regenerate this file.
//
// To regenerate:
//
//   $ cd <ngraph source dir>/test
//   $ ./update_dyn_replace_slice_reference.sh
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT EDIT THIS FILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!! THIS FILE IS AUTOGENERATED OUTSIDE OF THE BUILD PROCESS !!!!!!!!!!!!!!
//
// clang-format off

#include <algorithm>
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

struct DynReplaceSliceTestParamsBase
{
    bool success;
    element::Type input_element_type;
    element::Type replacement_element_type;
    Shape input_shape;
    Shape replacement_shape;
    vector<int64_t> lb_values;
    vector<int64_t> ub_values;
    vector<int64_t> strides_values;
    AxisSet lb_mask;
    AxisSet ub_mask;
    AxisSet new_mask;
    AxisSet shrink_mask;
    AxisSet ellipsis_mask;

    virtual ~DynReplaceSliceTestParamsBase() {}

    virtual void copy_input_values(const shared_ptr<runtime::Tensor>& input_tensor) = 0;
    virtual void copy_replacement_values(const shared_ptr<runtime::Tensor>& replacement_tensor) = 0;
    virtual void check_result_values(const std::shared_ptr<runtime::Tensor>& output_tensor) = 0;
};

template <typename Tinput,typename Treplacement>
struct DynReplaceSliceTestParams : public DynReplaceSliceTestParamsBase
{
    DynReplaceSliceTestParams(
        bool p_success,
        element::Type p_input_element_type,
        element::Type p_replacement_element_type,
        const Shape& p_input_shape,
        const Shape& p_replacement_shape,
        const vector<int64_t>& p_lb_values,
        const vector<int64_t>& p_ub_values,
        const vector<int64_t>& p_strides_values,
        const AxisSet& p_lb_mask,
        const AxisSet& p_ub_mask,
        const AxisSet& p_new_mask,
        const AxisSet& p_shrink_mask,
        const AxisSet& p_ellipsis_mask,
        const vector<Tinput>& p_expected_result_values,
        const vector<Treplacement>& p_replacement_values)
    {
        success = p_success;
        input_element_type = p_input_element_type;
        replacement_element_type = p_replacement_element_type;
        input_shape = p_input_shape;
        replacement_shape = p_replacement_shape;
        lb_values = p_lb_values;
        ub_values = p_ub_values;
        strides_values = p_strides_values;
        lb_mask = p_lb_mask;
        ub_mask = p_ub_mask;
        new_mask = p_new_mask;
        shrink_mask = p_shrink_mask;
        ellipsis_mask = p_ellipsis_mask;

        expected_result_values = p_expected_result_values;
        replacement_values = p_replacement_values;
    }

    vector<Tinput> expected_result_values;
    vector<Treplacement> replacement_values;

    virtual void copy_input_values(const shared_ptr<runtime::Tensor>& input_tensor) override
    {
        std::vector<Tinput> input_values(shape_size(input_shape));
        std::iota(input_values.begin(), input_values.end(), static_cast<Tinput>(0));
        copy_data(input_tensor, input_values);
    }

    virtual void copy_replacement_values(const shared_ptr<runtime::Tensor>& replacement_tensor) override
    {
        copy_data(replacement_tensor, replacement_values);
    }

    virtual void check_result_values(const std::shared_ptr<runtime::Tensor>& output_tensor) override
    {
        vector<Tinput> result_values = read_vector<Tinput>(output_tensor);
        EXPECT_EQ(result_values, expected_result_values);
    }
};

// We use a shared_ptr here because:
//  (1) we cannot use the objects directly, since DynReplaceSliceTestParamsBase is abstract;
//  (2) we cannot use references or raw pointers, since things won't get freed properly;
//  (3) we cannot use unique_ptr, since gtest requires a copy constructor.
struct DynReplaceSliceTest : ::testing::TestWithParam<shared_ptr<DynReplaceSliceTestParamsBase>>
{
};

NGRAPH_TEST_P(${BACKEND_NAME}, DynReplaceSliceTest, dyn_replace_slice)
{
    std::shared_ptr<DynReplaceSliceTestParamsBase> t = GetParam();

    auto backend = runtime::Backend::create("${BACKEND_NAME}",true);
    auto output = backend->create_dynamic_tensor(t->input_element_type, PartialShape::dynamic());

    auto setup = [&t, &backend, &output]() {
        auto arg = std::make_shared<op::Parameter>(t->input_element_type, t->input_shape);
        auto repl = std::make_shared<op::Parameter>(t->replacement_element_type, t->replacement_shape);
        auto lb = std::make_shared<op::Parameter>(element::i64, Shape{t->lb_values.size()});
        auto ub = std::make_shared<op::Parameter>(element::i64, Shape{t->ub_values.size()});
        auto strides = std::make_shared<op::Parameter>(element::i64, Shape{t->strides_values.size()});

        auto rsl = std::make_shared<op::DynReplaceSlice>(arg, repl,
                                                         lb, ub, strides,
                                                         t->lb_mask, t->ub_mask, t->new_mask,
                                                         t->shrink_mask, t->ellipsis_mask);

        auto f = std::make_shared<Function>(NodeVector{rsl}, ParameterVector{arg, repl, lb, ub, strides});

        auto ex = backend->compile(f);

        auto input_arg = backend->create_tensor(t->input_element_type, t->input_shape);
        auto input_repl = backend->create_tensor(t->replacement_element_type, t->replacement_shape);
        auto input_lb = backend->create_tensor(element::i64, Shape{t->lb_values.size()});
        auto input_ub = backend->create_tensor(element::i64, Shape{t->ub_values.size()});
        auto input_strides = backend->create_tensor(element::i64, Shape{t->strides_values.size()});
        t->copy_input_values(input_arg);
        t->copy_replacement_values(input_repl);
        copy_data(input_lb, t->lb_values);
        copy_data(input_ub, t->ub_values);
        copy_data(input_strides, t->strides_values);

        ex->call_with_validate({output}, {input_arg, input_repl, input_lb, input_ub, input_strides});
    };

    if (t->success)
    {
        setup();
        EXPECT_EQ(output->get_element_type(), t->input_element_type);
        EXPECT_EQ(output->get_shape(), t->input_shape);
        t->check_result_values(output);
    }
    else
    {
        EXPECT_ANY_THROW({
            setup();
        });
    }
}

NGRAPH_INSTANTIATE_TEST_CASE_P(${BACKEND_NAME},
                               dyn_replace_slice,
                               DynReplaceSliceTest,
                               (::testing::ValuesIn(
                                   std::vector<std::shared_ptr<DynReplaceSliceTestParamsBase>>{''')

def write_footer(f):
    f.write('''\
                                   })));
// clang-format on
''')


def main():
    if len(sys.argv) < 2:
        sys.stderr.write('Output filename is required\n')
        sys.exit(1)

    f = open(sys.argv[1], 'w')
    write_header(f)

    t = ReplaceSliceTestWriter(stream=f)

    t.set_shape((4,))
    for dt in ['int32','int64','float32','uint32']:
        t.set_dtype(dt)

        t[np.newaxis,3:0:-1] = None
        t[...] = None
        t[1:3] = None
        t[2] = None
        t[3:0:-2] = None
        t[3::-2] = None
        t[4::-2] = None
        t[5::-2] = None
        t[-9000:-8000:2] = None
        t[-9000:8000:2] = None
        t[-5:5:2] = None
        t[np.newaxis] = None
        t[np.newaxis,np.newaxis] = None
        t[np.newaxis,np.newaxis,...,np.newaxis] = None

        # Some tests with incorrect replacement shapes
        t[2] = np.ones(shape=(2,2), dtype=dt)

        t.set_shape((5,))
        t[3:0:-2] = None
        t[0:3:2] = None
        t[0:4:2] = None
        t[0:5:2] = None
        t[0:6:2] = None
        t[0:100:2] = None
        t[4:0:-2] = None
        t[4:0:-3] = None
        t[3:2:1] = None
        t[4::-2] = None

        #
        # A couple of tests for negative-stride slicing. The issue we want to
        # be on the lookout for is this:
        #
        #  [ORIGINAL]
        #   01234567
        #   ..1..0..   [5:0:-3]  # suppose we start with this, want to convert
        #    _____               # to pos stride. suppose that our stride is
        #                        # "uneven" wrt the slicing region, i.e. the
        #                        # start-to-end distance is not an even
        #                        # multiple of the strides (e.g. here: we get
        #                        # elements 5 and 2.)
        #
        #  [INCORRECT]
        #   01234567
        #   .0..1...   [1:6:3]   # if we just reverse the sign of the stride
        #    _____               # and flip the start/end indices while
        #                        # traversing, we will get out the wrong
        #                        # elements. (e.g. here: we get elements 1 and
        #                        # 4, which are not what we want.)
        #
        #  [CORRECT]
        #   01234567
        #   ..0..1..   [2:6:3]   # the correct thing to do is to adjust the
        #     ____               # start of our reversed slice to be the last
        #                        # element that is *actually* touched by the
        #                        # original negative striding, not the
        #                        # boundary of the region. (e.g. here: we get
        #                        # elements 2 and 5, which are what we want.)
        #
        # There's some logic to do this transformation in DynElimination, but
        # it feels a bit delicate.
        #
        t.set_shape((8,))
        t[5:2:-3] = None
        t[5:1:-3] = None
        t[5:0:-3] = None
        t[5::-3] = None
        t[6:3:-3] = None
        t[6:2:-3] = None
        t[6:1:-3] = None
        t[6::-3] = None
        t[7:1:-3] = None
        t[7:0:-3] = None
        t[7::-3] = None

    t.set_dtype('int32')
    t.set_shape((4, 5))
    t[2:4, ...] = None
    t[4:2, ...] = None
    t[4:2:-3, ...] = None
    t[-100:100, ...] = None
    t[..., 2:] = None
    t[..., 2:4] = None
    t[..., :] = None
    t[..., -100:100] = None
    t.set_shape((5, 6, 10, 8))
    t[2:4, ..., 1:7:3, 7:2:-2] = None
    t[..., 1:7:3, 7:2:-2] = None
    t[2:4, ..., :3, -3:2:-2] = None
    t[2:4, ..., 1:7:-3, 7:2:-2] = None
    t[2:4, ..., :, np.newaxis, 0] = None

    t.set_shape((2, 2, 3, 2, 3, 3))
    t[2:6:2, ..., :, 2:1:-1] = None
    t[np.newaxis, 1, ..., np.newaxis, 2:1:-1] = None
    t[1, ..., np.newaxis, 2:1:-1] = None
    t[np.newaxis, np.newaxis, 2:1:-1, ...] = None

    t.set_shape((3, 3, 3, 2, 3))
    t[6:1:-2, ..., 1:2, 2:1:-1] = None

    t.set_shape((3, 3, 3, 2, 3))
    t[..., 1:2, 2:1:-1] = None

    t.set_dtype('int32')
    t[80000] = None # error expected (shrink-axis OOB)
    t[-80000] = None # error expected (shrink-axis OOB)
    t[:,:] = None # error expected (too many indices)
    t[0:0:0] = None # error expected (stride==0)
    t[0:1:0] = None # error expected (stride==0)
    t[0:2:0] = None # error expected (stride==0)
    t[::0] = None # error expected (stride==0)

    t.set_shape((2,3,4))

    t.set_dtype('int32')
    # Test with incorrect DT
    t[...] = np.ones(shape=(2,3,4), dtype='float32')
    # Test some cases where auto-broadcast would be required
    t[...] = np.ones(shape=(1,3,4), dtype='int32')
    t[...] = np.ones(shape=(3,4), dtype='int32')
    t[0,...,0] = np.ones(shape=(1), dtype='int32')

    for dt in ['int32','int64','float32','uint32']:
        t.set_dtype(dt)

        t[1,np.newaxis] = None
        t[-1,-1,np.newaxis] = None

    t.set_shape((2,4,6,8,2,2,2))
    for dt in ['int32','int64','float32','uint32']:
        t.set_dtype(dt)
        t[0:,:4,2:6:2,7:3:-2,np.newaxis,...,1] = None

    t.set_dtype('int32')
    t[...,...] = None # error expected (too many ellipses)

    write_footer(f)
    f.close()

if __name__ == "__main__":
    main()
