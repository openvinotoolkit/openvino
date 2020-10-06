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

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_vector)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{6, 6, 6, 6}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_matrix)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1}),
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{6, 6, 6, 6}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_tensor)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1, 2}),
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{6, 6, 6, 6, 6, 6, 6, 6}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_trivial)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f =
        make_shared<Function>(make_shared<op::Broadcast>(A, shape, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 6, 8, 16, 32, 64, 128});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 4, 6, 8, 16, 32, 64, 128}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_vector_colwise)
{
    Shape shape_a{3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{1}),
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_vector_rowwise)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

// Test hybrid mechanism after broadcast
NGRAPH_TEST(${BACKEND_NAME}, broadcast_vector_rowwise_reversed)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 4};
    auto broadcast = make_shared<op::Broadcast>(A, shape_r, AxisSet{0});
    auto reverse = make_shared<op::Reverse>(broadcast, AxisSet{1});
    auto f = make_shared<Function>(reverse, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_vector_rowwise_int64)
{
    Shape shape_a{4};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_r{3, 4};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::i64, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int64_t>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}), read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_to_matrix_int64)
{
    Shape shape_a{1};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_r{3, 1};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{4});
    auto result = backend->create_tensor(element::i64, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int64_t>{4, 4, 4}), read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_scalar_to_matrix_int32)
{
    Shape shape_a{1};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_r{3, 1};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{4});
    auto result = backend->create_tensor(element::i32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{4, 4, 4}), read_vector<int32_t>(result));
}

static void broadcast_test_helper(const Shape& shape_a, const Shape& shape_r, const AxisSet& axis)
{
    auto A = make_shared<op::Parameter>(element::f32, shape_a);

    vector<float> inp_data(shape_size<const Shape>(shape_a));
    iota(inp_data.begin(), inp_data.end(), 1.f);

    auto f =
        make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, axis), ParameterVector{A});

    auto ref_backend = runtime::Backend::create("INTERPRETER");
    auto wrk_backend = runtime::Backend::create("${BACKEND_NAME}");

    auto wrk_a = wrk_backend->create_tensor(element::f32, shape_a);
    copy_data(wrk_a, inp_data);

    auto ref_a = ref_backend->create_tensor(element::f32, shape_a);
    copy_data(ref_a, inp_data);

    auto wrk_result = wrk_backend->create_tensor(element::f32, shape_r);
    auto ref_result = ref_backend->create_tensor(element::f32, shape_r);

    auto wrk_handle = wrk_backend->compile(f);
    auto ref_handle = ref_backend->compile(f);
    wrk_handle->call_with_validate({wrk_result}, {wrk_a});
    ref_handle->call_with_validate({ref_result}, {ref_a});
    EXPECT_TRUE(test::all_close_f(
        read_vector<float>(ref_result), read_vector<float>(wrk_result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_algo_vector_middle)
{
    Shape shape_a{2};
    Shape shape_r{3, 2, 4};
    AxisSet axis{0, 2};
    broadcast_test_helper(shape_a, shape_r, axis);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_algo_vector_forward_2)
{
    Shape shape_a{2};
    Shape shape_r{3, 2};
    AxisSet axis{0};
    broadcast_test_helper(shape_a, shape_r, axis);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_algo_vector_forward_3)
{
    Shape shape_a{2};
    Shape shape_r{4, 3, 2};
    AxisSet axis{0, 1};
    broadcast_test_helper(shape_a, shape_r, axis);
}
NGRAPH_TEST(${BACKEND_NAME}, broadcast_algo_vector_forward_4)
{
    Shape shape_a{2};
    Shape shape_r{5, 4, 3, 2};
    AxisSet axis{0, 1, 2};
    broadcast_test_helper(shape_a, shape_r, axis);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_algo_scalar)
{
    Shape shape_a{};
    Shape shape_r{5, 4, 3, 2};
    AxisSet axis{0, 1, 2, 3};
    broadcast_test_helper(shape_a, shape_r, axis);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_algo_vector_backward_2)
{
    Shape shape_a{2};
    Shape shape_r{2, 3};
    AxisSet axis{1};
    broadcast_test_helper(shape_a, shape_r, axis);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_algo_vector_backward_3)
{
    Shape shape_a{2};
    Shape shape_r{2, 3, 4};
    AxisSet axis{1, 2};
    broadcast_test_helper(shape_a, shape_r, axis);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_algo_vector_backward_4)
{
    Shape shape_a{2};
    Shape shape_r{2, 3, 4, 5};
    AxisSet axis{1, 2, 3};
    broadcast_test_helper(shape_a, shape_r, axis);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_algo_matrix_backward_4)
{
    Shape shape_a{4, 5};
    Shape shape_r{2, 3, 4, 5};
    AxisSet axis{0, 1};
    broadcast_test_helper(shape_a, shape_r, axis);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_algo_matrix_stride_1)
{
    Shape shape_a{3, 5};
    Shape shape_r{2, 3, 4, 5};
    AxisSet axis{0, 2};
    broadcast_test_helper(shape_a, shape_r, axis);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_algo_matrix_stride_2)
{
    Shape shape_a{3, 4};
    Shape shape_r{2, 3, 4, 5};
    AxisSet axis{0, 3};
    broadcast_test_helper(shape_a, shape_r, axis);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_algo_matrix_stride_3)
{
    Shape shape_a{2, 4};
    Shape shape_r{2, 3, 4, 5};
    AxisSet axis{1, 3};
    broadcast_test_helper(shape_a, shape_r, axis);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_algo_3d_backward)
{
    Shape shape_a{2, 3, 4};
    Shape shape_r{5, 2, 3, 4};
    AxisSet axis{0};
    broadcast_test_helper(shape_a, shape_r, axis);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_algo_3d_stride_1)
{
    Shape shape_a{2, 3, 4};
    Shape shape_r{2, 5, 3, 4};
    AxisSet axis{1};
    broadcast_test_helper(shape_a, shape_r, axis);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_algo_3d_stride_2)
{
    Shape shape_a{2, 3, 4};
    Shape shape_r{2, 3, 5, 4};
    AxisSet axis{2};
    broadcast_test_helper(shape_a, shape_r, axis);
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_matrix_0)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}),
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 2, 3, 4, 1, 2, 3, 4}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_matrix_1)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{1}),
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 2, 1, 2, 3, 4, 3, 4}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, broadcast_matrix_2)
{
    Shape shape_a{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto f = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{2}),
                                   ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 1, 2, 2, 3, 3, 4, 4}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}
