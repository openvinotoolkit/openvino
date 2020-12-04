//*****************************************************************************
// Copyright 2020 Intel Corporation
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

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, log_softmax_1d_single_value)
{
    Shape shape{1};
    auto A = make_shared<op::Parameter>(element::f32, shape);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1});
    auto result = backend->create_tensor(element::f32, shape);

    std::vector<float> expected_result{0};

    auto f = make_shared<Function>(make_shared<op::v5::LogSoftmax>(A, 0), ParameterVector{A});
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, log_softmax_2d_axis0)
{
    Shape shape{2, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 1, 2, 3, 10000, 10001, 10002, 10003});
    auto result = backend->create_tensor(element::f32, shape);

    std::vector<float> expected_result{-10000., -10000., -10000., -10000., 0., 0., 0., 0.};

    auto f = make_shared<Function>(make_shared<op::v5::LogSoftmax>(A, 0), ParameterVector{A});
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, log_softmax_2d_axis1)
{
    Shape shape{2, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 1, 2, 3, 10000, 10001, 10002, 10003});
    auto result = backend->create_tensor(element::f32, shape);

    std::vector<float> expected_result{-3.4401896,
                                       -2.4401896,
                                       -1.4401897,
                                       -0.4401897,
                                       -3.4401896,
                                       -2.4401896,
                                       -1.4401897,
                                       -0.4401897};

    auto f = make_shared<Function>(make_shared<op::v5::LogSoftmax>(A, 1), ParameterVector{A});
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, log_softmax_2d_axis_neg1)
{
    Shape shape{2, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 1, 2, 3, 10000, 10001, 10002, 10003});
    auto result = backend->create_tensor(element::f32, shape);

    std::vector<float> expected_result{-3.4401896,
                                       -2.4401896,
                                       -1.4401897,
                                       -0.4401897,
                                       -3.4401896,
                                       -2.4401896,
                                       -1.4401897,
                                       -0.4401897};

    auto f = make_shared<Function>(make_shared<op::v5::LogSoftmax>(A, -1), ParameterVector{A});
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, log_softmax_2d_axis_neg2)
{
    Shape shape{2, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 1, 2, 3, 10000, 10001, 10002, 10003});
    auto result = backend->create_tensor(element::f32, shape);

    std::vector<float> expected_result{-10000., -10000., -10000., -10000., 0., 0., 0., 0.};

    auto f = make_shared<Function>(make_shared<op::v5::LogSoftmax>(A, -2), ParameterVector{A});
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, log_softmax_3d_axis_0)
{
    Shape shape{3, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, shape);

    std::vector<float> expected_result{-12.0024818,
                                       -12.0024818,
                                       -12.0024818,
                                       -12.0024818,
                                       -12.0024818,
                                       -12.0024818,
                                       -6.00248181,
                                       -6.00248181,
                                       -6.00248181,
                                       -6.00248181,
                                       -6.00248181,
                                       -6.00248181,
                                       -2.48181414e-03,
                                       -2.48181414e-03,
                                       -2.48181414e-03,
                                       -2.48181414e-03,
                                       -2.48181414e-03,
                                       -2.48181414e-03};

    auto f = make_shared<Function>(make_shared<op::v5::LogSoftmax>(A, 0), ParameterVector{A});
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, log_softmax_3d_axis_1)
{
    Shape shape{3, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, shape);

    std::vector<float> expected_result{-3.04858735,
                                       -3.04858735,
                                       -3.04858735,
                                       -0.04858735,
                                       -0.04858735,
                                       -0.04858735,
                                       -3.04858735,
                                       -3.04858735,
                                       -3.04858735,
                                       -0.04858735,
                                       -0.04858735,
                                       -0.04858735,
                                       -3.04858735,
                                       -3.04858735,
                                       -3.04858735,
                                       -0.04858735,
                                       -0.04858735,
                                       -0.04858735};

    auto f = make_shared<Function>(make_shared<op::v5::LogSoftmax>(A, 1), ParameterVector{A});
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, log_softmax_3d_axis_2)
{
    Shape shape{3, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, shape);

    std::vector<float> expected_result{-2.40760596,
                                       -1.40760596,
                                       -0.40760596,
                                       -2.40760596,
                                       -1.40760596,
                                       -0.40760596,
                                       -2.40760596,
                                       -1.40760596,
                                       -0.40760596,
                                       -2.40760596,
                                       -1.40760596,
                                       -0.40760596,
                                       -2.40760596,
                                       -1.40760596,
                                       -0.40760596,
                                       -2.40760596,
                                       -1.40760596,
                                       -0.40760596};

    auto f = make_shared<Function>(make_shared<op::v5::LogSoftmax>(A, 2), ParameterVector{A});
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, log_softmax_3d_axis_neg1)
{
    Shape shape{3, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, shape);

    std::vector<float> expected_result{-2.40760596,
                                       -1.40760596,
                                       -0.40760596,
                                       -2.40760596,
                                       -1.40760596,
                                       -0.40760596,
                                       -2.40760596,
                                       -1.40760596,
                                       -0.40760596,
                                       -2.40760596,
                                       -1.40760596,
                                       -0.40760596,
                                       -2.40760596,
                                       -1.40760596,
                                       -0.40760596,
                                       -2.40760596,
                                       -1.40760596,
                                       -0.40760596};

    auto f = make_shared<Function>(make_shared<op::v5::LogSoftmax>(A, -1), ParameterVector{A});
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, log_softmax_3d_axis_neg2)
{
    Shape shape{3, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, shape);

    std::vector<float> expected_result{-3.04858735,
                                       -3.04858735,
                                       -3.04858735,
                                       -0.04858735,
                                       -0.04858735,
                                       -0.04858735,
                                       -3.04858735,
                                       -3.04858735,
                                       -3.04858735,
                                       -0.04858735,
                                       -0.04858735,
                                       -0.04858735,
                                       -3.04858735,
                                       -3.04858735,
                                       -3.04858735,
                                       -0.04858735,
                                       -0.04858735,
                                       -0.04858735};

    auto f = make_shared<Function>(make_shared<op::v5::LogSoftmax>(A, -2), ParameterVector{A});
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, log_softmax_3d_axis_neg3)
{
    Shape shape{3, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, shape);

    std::vector<float> expected_result{-12.0024818,
                                       -12.0024818,
                                       -12.0024818,
                                       -12.0024818,
                                       -12.0024818,
                                       -12.0024818,
                                       -6.00248181,
                                       -6.00248181,
                                       -6.00248181,
                                       -6.00248181,
                                       -6.00248181,
                                       -6.00248181,
                                       -2.48181414e-03,
                                       -2.48181414e-03,
                                       -2.48181414e-03,
                                       -2.48181414e-03,
                                       -2.48181414e-03,
                                       -2.48181414e-03};

    auto f = make_shared<Function>(make_shared<op::v5::LogSoftmax>(A, -3), ParameterVector{A});
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close(expected_result, read_vector<float>(result)));
}
