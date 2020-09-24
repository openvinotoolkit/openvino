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
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, quantized_dot_u8u8)
{
    Shape shape_a{1, 2}; // input shape
    vector<uint8_t> a_data = {2, 3};
    Shape shape_b{2, 3}; // filter shape
    vector<uint8_t> b_data = {0, 2, 4, 1, 3, 5};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto B = make_shared<op::Parameter>(element::u8, shape_b);
    auto input_scale = op::Constant::create(element::f32, Shape{}, {2});
    auto input_zero_point = op::Constant::create(element::u8, Shape{}, {0});
    auto filter_scale = op::Constant::create(element::f32, Shape{}, {1});
    auto filter_zero_point = op::Constant::create(element::u8, Shape{}, {0});
    auto output_scale = op::Constant::create(element::f32, Shape{}, {2});
    auto output_zero_point = op::Constant::create(element::u8, Shape{}, {0});
    AxisSet axes{};

    Shape shape_r{1, 3}; // output shape
    auto QD = make_shared<op::QuantizedDot>(A,
                                            B,
                                            1,
                                            input_scale,
                                            input_zero_point,
                                            filter_scale,
                                            filter_zero_point,
                                            output_scale,
                                            output_zero_point,
                                            element::u8,
                                            axes,
                                            axes,
                                            axes);
    auto f = make_shared<Function>(NodeVector{QD}, ParameterVector{A, B});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::u8, shape_b);
    copy_data(b, b_data);
    auto result = backend->create_tensor(element::u8, shape_r);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<uint8_t>{3, 13, 23}), read_vector<uint8_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, quantized_dot_int32_output)
{
    Shape shape_a{1, 2}; // input shape
    vector<uint8_t> a_data = {2, 3};
    Shape shape_b{2, 3}; // filter shape
    vector<int8_t> b_data = {0, 1, 2, 3, 4, 5};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto B = make_shared<op::Parameter>(element::i8, shape_b);
    auto input_scale = op::Constant::create(element::f32, Shape{}, {1});
    auto input_zero_point = op::Constant::create(element::u8, Shape{}, {0});
    auto filter_scale = op::Constant::create(element::f32, Shape{}, {1});
    auto filter_zero_point = op::Constant::create(element::i8, Shape{}, {0});
    auto output_scale = op::Constant::create(element::f32, Shape{}, {1});
    auto output_zero_point = op::Constant::create(element::i32, Shape{}, {0});
    AxisSet axes{};

    Shape shape_r{1, 3}; // output shape
    auto QD = make_shared<op::QuantizedDot>(A,
                                            B,
                                            1,
                                            input_scale,
                                            input_zero_point,
                                            filter_scale,
                                            filter_zero_point,
                                            output_scale,
                                            output_zero_point,
                                            element::i32,
                                            axes,
                                            axes,
                                            axes);
    auto f = make_shared<Function>(NodeVector{QD}, ParameterVector{A, B});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::i8, shape_b);
    copy_data(b, b_data);
    auto result = backend->create_tensor(element::i32, shape_r);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{9, 14, 19}), read_vector<int32_t>(result));
}
