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

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_3d)
{
    Shape shape{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Softmax>(A, 0), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-10, -20, -30, -40, -50, -60, -1, -2, -3, -4, -5, -6});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(-10) + expf(-1);
    auto d1 = expf(-20) + expf(-2);
    auto d2 = expf(-30) + expf(-3);
    auto d3 = expf(-40) + expf(-4);
    auto d4 = expf(-50) + expf(-5);
    auto d5 = expf(-60) + expf(-6);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    vector<float> expected{expf(-10) / d0,
                           expf(-20) / d1,
                           expf(-30) / d2,
                           expf(-40) / d3,
                           expf(-50) / d4,
                           expf(-60) / d5,
                           expf(-1) / d0,
                           expf(-2) / d1,
                           expf(-3) / d2,
                           expf(-4) / d3,
                           expf(-5) / d4,
                           expf(-6) / d5};

    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_3d_double)
{
    Shape shape{2, 2, 3};
    auto A = make_shared<op::Parameter>(element::f64, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Softmax>(A, 0), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f64, shape);
    copy_data(a, vector<double>{-10, -20, -30, -40, -50, -60, -1, -2, -3, -4, -5, -6});
    auto result = backend->create_tensor(element::f64, shape);

    auto d0 = exp(-10) + exp(-1);
    auto d1 = exp(-20) + exp(-2);
    auto d2 = exp(-30) + exp(-3);
    auto d3 = exp(-40) + exp(-4);
    auto d4 = exp(-50) + exp(-5);
    auto d5 = exp(-60) + exp(-6);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    vector<double> expected{exp(-10) / d0,
                            exp(-20) / d1,
                            exp(-30) / d2,
                            exp(-40) / d3,
                            exp(-50) / d4,
                            exp(-60) / d5,
                            exp(-1) / d0,
                            exp(-2) / d1,
                            exp(-3) / d2,
                            exp(-4) / d3,
                            exp(-5) / d4,
                            exp(-6) / d5};

    EXPECT_TRUE(test::all_close(expected, read_vector<double>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_2d_axis_1)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Softmax>(A, 1), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-10, -20, -30, -40, -50, -60});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(-10) + expf(-20) + expf(-30);
    auto d1 = expf(-40) + expf(-50) + expf(-60);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    vector<float> expected{expf(-10) / d0,
                           expf(-20) / d0,
                           expf(-30) / d0,
                           expf(-40) / d1,
                           expf(-50) / d1,
                           expf(-60) / d1};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_2d_axis_0)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Softmax>(A, 0), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-10, -20, -30, -40, -50, -60});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(-10) + expf(-40);
    auto d1 = expf(-20) + expf(-50);
    auto d2 = expf(-30) + expf(-60);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    vector<float> expected{expf(-10) / d0,
                           expf(-20) / d1,
                           expf(-30) / d2,
                           expf(-40) / d0,
                           expf(-50) / d1,
                           expf(-60) / d2};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_axis_3d_trivial)
{
    Shape shape{1, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Softmax>(A, 0), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-10, -20, -30, -40, -50, -60});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    vector<float> expected{1, 1, 1, 1, 1, 1};
    EXPECT_TRUE(test::all_close(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_underflow)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Softmax>(A, 0), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto low = std::numeric_limits<float>::lowest();

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{low, 1, 2, 3, 4, 5});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(low) + expf(3);
    auto d1 = expf(1) + expf(4);
    auto d2 = expf(2) + expf(5);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    vector<float> expected{
        expf(low) / d0, expf(1) / d1, expf(2) / d2, expf(3) / d0, expf(4) / d1, expf(5) / d2};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, softmax_overflow)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Softmax>(A, 0), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto high = std::numeric_limits<float>::max();

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{high, 1, 2, 3, 4, 5});
    auto result = backend->create_tensor(element::f32, shape);

    auto d0 = expf(high - high) + expf(3 - high);
    auto d1 = expf(1) + expf(4);
    auto d2 = expf(2) + expf(5);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    vector<float> expected{expf(high - high) / d0,
                           expf(1) / d1,
                           expf(2) / d2,
                           expf(3 - high) / d0,
                           expf(4) / d1,
                           expf(5) / d2};
    EXPECT_TRUE(test::all_close_f(expected, read_vector<float>(result)));
}
