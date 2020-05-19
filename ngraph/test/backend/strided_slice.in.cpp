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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close_f.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

template <typename T>
void check_strided_slice_success(const element::Type& input_element_type,
                                 const Shape& input_shape,
                                 const std::vector<int64_t>& begin_values,
                                 const std::vector<int64_t>& end_values,
                                 const std::vector<int64_t>& strides_values,
                                 const std::vector<int64_t>& begin_mask,
                                 const std::vector<int64_t>& end_mask,
                                 const std::vector<int64_t>& new_axis_mask,
                                 const std::vector<int64_t>& shrink_axis_mask,
                                 const std::vector<int64_t>& ellipsis_mask,
                                 const Shape& expected_output_shape,
                                 const std::vector<T>& expected_values)
{
    auto arg = std::make_shared<op::Parameter>(input_element_type, input_shape);
    auto begin_op = make_shared<ngraph::op::Parameter>(element::i64, Shape{begin_values.size()});
    auto end_op = make_shared<ngraph::op::Parameter>(element::i64, Shape{end_values.size()});
    auto strides_op =
        make_shared<ngraph::op::Parameter>(element::i64, Shape{strides_values.size()});

    std::vector<T> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), static_cast<T>(0));

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(arg,
                                                                begin_op,
                                                                end_op,
                                                                strides_op,
                                                                begin_mask,
                                                                end_mask,
                                                                new_axis_mask,
                                                                shrink_axis_mask,
                                                                ellipsis_mask);

    auto f = std::make_shared<Function>(NodeVector{strided_slice},
                                        ParameterVector{arg, begin_op, end_op, strides_op});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    auto ex = backend->compile(f);

    auto arg_tensor = backend->create_tensor(input_element_type, input_shape);
    auto begin_tensor = backend->create_tensor(element::i64, Shape{begin_values.size()});
    auto end_tensor = backend->create_tensor(element::i64, Shape{end_values.size()});
    auto strides_tensor = backend->create_tensor(element::i64, Shape{strides_values.size()});
    copy_data(arg_tensor, input_values);
    copy_data(begin_tensor, begin_values);
    copy_data(end_tensor, end_values);
    copy_data(strides_tensor, strides_values);

    auto output = backend->create_dynamic_tensor(input_element_type, PartialShape::dynamic());

    ex->call_with_validate({output}, {arg_tensor, begin_tensor, end_tensor, strides_tensor});

    EXPECT_EQ(output->get_element_type(), input_element_type);
    EXPECT_EQ(output->get_shape(), expected_output_shape);

    auto output_values = read_vector<T>(output);

    EXPECT_EQ(output_values, expected_values);
}

template <typename T>
void check_strided_slice_stride_optional_success(const element::Type& input_element_type,
                                                 const Shape& input_shape,
                                                 const std::vector<int64_t>& begin_values,
                                                 const std::vector<int64_t>& end_values,
                                                 const std::vector<int64_t>& begin_mask,
                                                 const std::vector<int64_t>& end_mask,
                                                 const std::vector<int64_t>& new_axis_mask,
                                                 const std::vector<int64_t>& shrink_axis_mask,
                                                 const std::vector<int64_t>& ellipsis_mask,
                                                 const Shape& expected_output_shape,
                                                 const std::vector<T>& expected_values)
{
    auto arg = std::make_shared<op::Parameter>(input_element_type, input_shape);
    auto begin_op = make_shared<ngraph::op::Parameter>(element::i64, Shape{begin_values.size()});
    auto end_op = make_shared<ngraph::op::Parameter>(element::i64, Shape{end_values.size()});

    std::vector<T> input_values(shape_size(input_shape));
    std::iota(input_values.begin(), input_values.end(), static_cast<T>(0));

    auto strided_slice = std::make_shared<op::v1::StridedSlice>(arg,
                                                                begin_op,
                                                                end_op,
                                                                begin_mask,
                                                                end_mask,
                                                                new_axis_mask,
                                                                shrink_axis_mask,
                                                                ellipsis_mask);

    auto f = std::make_shared<Function>(NodeVector{strided_slice},
                                        ParameterVector{arg, begin_op, end_op});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);
    auto ex = backend->compile(f);

    auto arg_tensor = backend->create_tensor(input_element_type, input_shape);
    auto begin_tensor = backend->create_tensor(element::i64, Shape{begin_values.size()});
    auto end_tensor = backend->create_tensor(element::i64, Shape{end_values.size()});
    copy_data(arg_tensor, input_values);
    copy_data(begin_tensor, begin_values);
    copy_data(end_tensor, end_values);

    auto output = backend->create_dynamic_tensor(input_element_type, PartialShape::dynamic());

    ex->call_with_validate({output}, {arg_tensor, begin_tensor, end_tensor});

    EXPECT_EQ(output->get_element_type(), input_element_type);
    EXPECT_EQ(output->get_shape(), expected_output_shape);

    auto output_values = read_vector<T>(output);

    EXPECT_EQ(output_values, expected_values);
}

// slices are: [1,newaxis]
// dtype is: uint32
// input shape is: Shape{2,3,4}
// expected output shape is Shape{1,3,4}
NGRAPH_TEST(${BACKEND_NAME}, strided_slice_0)
{
    check_strided_slice_success<uint32_t>(
        element::u32,
        Shape{2, 3, 4},
        std::vector<int64_t>{1, 0},
        std::vector<int64_t>{0, 0},
        std::vector<int64_t>{1, 1},
        std::vector<int64_t>{0, 0, 0},
        std::vector<int64_t>{0, 0, 0},
        std::vector<int64_t>{0, 1, 0},
        std::vector<int64_t>{1, 0, 0},
        std::vector<int64_t>{0, 0, 0},
        Shape{1, 3, 4},
        std::vector<uint32_t>{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
}

// slices are: [0:,:4,2:6:2,7:3:-2,newaxis,...,1]
// dtype is: uint32
// input shape is: Shape{2,4,6,8,2,2,2}
// expected output shape is Shape{2,4,2,2,1,2,2}
NGRAPH_TEST(${BACKEND_NAME}, strided_slice_1)
{
    check_strided_slice_success<uint32_t>(
        element::u32,
        Shape{2, 4, 6, 8, 2, 2, 2},
        std::vector<int64_t>{0, 0, 2, 7, 0, 0, 1},
        std::vector<int64_t>{0, 4, 6, 3, 0, 0, 0},
        std::vector<int64_t>{1, 1, 2, -2, 1, 1, 1},
        std::vector<int64_t>{0, 1, 0, 0, 0, 0, 0},
        std::vector<int64_t>{1, 0, 0, 0, 0, 0, 0},
        std::vector<int64_t>{0, 0, 0, 0, 1, 0, 0},
        std::vector<int64_t>{0, 0, 0, 0, 0, 0, 1},
        std::vector<int64_t>{0, 0, 0, 0, 0, 1, 0},
        Shape{2, 4, 2, 2, 1, 2, 2},
        std::vector<uint32_t>{
            185,  187,  189,  191,  169,  171,  173,  175,  313,  315,  317,  319,  297,
            299,  301,  303,  569,  571,  573,  575,  553,  555,  557,  559,  697,  699,
            701,  703,  681,  683,  685,  687,  953,  955,  957,  959,  937,  939,  941,
            943,  1081, 1083, 1085, 1087, 1065, 1067, 1069, 1071, 1337, 1339, 1341, 1343,
            1321, 1323, 1325, 1327, 1465, 1467, 1469, 1471, 1449, 1451, 1453, 1455, 1721,
            1723, 1725, 1727, 1705, 1707, 1709, 1711, 1849, 1851, 1853, 1855, 1833, 1835,
            1837, 1839, 2105, 2107, 2109, 2111, 2089, 2091, 2093, 2095, 2233, 2235, 2237,
            2239, 2217, 2219, 2221, 2223, 2489, 2491, 2493, 2495, 2473, 2475, 2477, 2479,
            2617, 2619, 2621, 2623, 2601, 2603, 2605, 2607, 2873, 2875, 2877, 2879, 2857,
            2859, 2861, 2863, 3001, 3003, 3005, 3007, 2985, 2987, 2989, 2991});
}

// slices are: [-1,-1,newaxis]
// dtype is: uint32
// input shape is: Shape{2,3,4}
// expected output shape is Shape{1,4}
NGRAPH_TEST(${BACKEND_NAME}, strided_slice_stride_optional)
{
    check_strided_slice_stride_optional_success<uint32_t>(element::u32,
                                                          Shape{2, 3, 4},
                                                          std::vector<int64_t>{-1, -1, 0},
                                                          std::vector<int64_t>{0, 0, 0},
                                                          std::vector<int64_t>{0, 0, 0},
                                                          std::vector<int64_t>{0, 0, 0},
                                                          std::vector<int64_t>{0, 0, 1},
                                                          std::vector<int64_t>{1, 1, 0},
                                                          std::vector<int64_t>{0, 0, 0},
                                                          Shape{1, 4},
                                                          std::vector<uint32_t>{20, 21, 22, 23});
}
