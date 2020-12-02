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
#include "op/convolution.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, convolution_outlining)
{
    Shape shape_a{1, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape_a);
    Shape shape_b{2, 2, 1, 1};
    auto B = make_shared<op::Parameter>(element::Type_t::f32, shape_b);
    Shape shape_r{1, 2, 2, 2};
    auto conv1 = make_shared<op::v0::Convolution>(A,
                                                  B,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
    auto conv2 = make_shared<op::v0::Convolution>(conv1,
                                                  B,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
    auto f = make_shared<Function>(conv2, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape_a);
    copy_data(a, vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    auto b = backend->create_tensor(element::Type_t::f32, shape_b);
    copy_data(b, vector<float>{1.0f, 1.0f, 1.0f, 1.0f});
    auto result = backend->create_tensor(element::Type_t::f32, shape_r);

    vector<float> expected_result{4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(vector<float>{expected_result}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_simple)
{
    Shape shape_a{1, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape_a);
    Shape shape_b{2, 2, 1, 1};
    auto B = make_shared<op::Parameter>(element::Type_t::f32, shape_b);
    Shape shape_r{1, 2, 2, 2};
    auto conv1 = make_shared<op::v0::Convolution>(A,
                                                  B,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});

    auto f = make_shared<Function>(conv1, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape_a);
    copy_data(a, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    auto b = backend->create_tensor(element::Type_t::f32, shape_b);
    copy_data(b, vector<float>{3.0f, 3.0f, 3.0f, 3.0f});
    auto result = backend->create_tensor(element::Type_t::f32, shape_r);

    vector<float> expected_result{18.0f, 24.0f, 30.0f, 36.0f, 18.0f, 24.0f, 30.0f, 36.0f};

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(vector<float>{expected_result}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, convolution_simple_padding)
{
    Shape shape_a{1, 1, 2, 2};
    auto A = make_shared<op::Parameter>(element::Type_t::f32, shape_a);
    Shape shape_b{1, 1, 1, 1};
    auto B = make_shared<op::Parameter>(element::Type_t::f32, shape_b);
    Shape shape_r{1, 1, 5, 5};
    auto conv1 = make_shared<op::v0::Convolution>(A,
                                                  B,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{1, 1},
                                                  CoordinateDiff{2, 2},
                                                  Strides{1, 1});

    auto f = make_shared<Function>(conv1, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape_a);
    copy_data(a, vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    auto b = backend->create_tensor(element::Type_t::f32, shape_b);
    copy_data(b, vector<float>{2.0f});
    auto result = backend->create_tensor(element::Type_t::f32, shape_r);
    // clang-format off
    vector<float> expected_result{0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 2.0f, 4.0f, 0.0f, 0.0f,
                                  0.0f, 6.0f, 8.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // clang-format on
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(vector<float>{expected_result}, read_vector<float>(result)));
}

// The purpose of this test is to check if we can allow
// data_batch_shape as a node rather than argument
NGRAPH_TEST(${BACKEND_NAME}, dyn_convolution_backprop_data)
{
    Shape shape_filter{6, 3, 3, 3};
    auto filters = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    Shape shape_delta{2, 6, 3, 3};
    auto deltas = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    Shape shape_data_batch_shape{2, 3, 5, 5};
    auto data_batch_shape =
        make_shared<op::Parameter>(element::Type_t::i64, PartialShape{Dimension::dynamic()});
    auto strides = Strides{1, 1};
    auto dilations = Strides{1, 1};
    auto padding_begin = CoordinateDiff{0, 0};
    auto padding_end = CoordinateDiff{0, 0};

    auto conv1 = make_shared<op::v1::ConvolutionBackpropData>(
        deltas, filters, data_batch_shape, strides, padding_begin, padding_end, dilations);

    auto f = make_shared<Function>(conv1, ParameterVector{deltas, filters, data_batch_shape});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    auto handle = backend->compile(f);

    auto result = backend->create_dynamic_tensor(element::Type_t::f32, PartialShape::dynamic());

    vector<float> filter, delta, expected_result;

    for (int i = 0; i < 6 * 3 * 3 * 3; i++)
        filter.emplace_back(i);

    for (int i = 0; i < 2 * 6 * 3 * 3; i++)
        delta.emplace_back(i);

    for (int i = 0; i < 2 * 3 * 5 * 5; i++)
        expected_result.emplace_back(i);

    vector<int64_t> shapes = {5, 5};

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::Type_t::f32, shape_delta);
    copy_data(a, delta);
    auto b = backend->create_tensor(element::Type_t::f32, shape_filter);
    copy_data(b, filter);
    auto c = backend->create_tensor(element::Type_t::i64,
                                    Shape{shapes.size()}); // dynamic data batch shape
    copy_data(c, shapes);
    handle->call_with_validate({result}, {a, b, c});
    EXPECT_FALSE(test::all_close_f(vector<float>{expected_result}, read_vector<float>(result)));
}
