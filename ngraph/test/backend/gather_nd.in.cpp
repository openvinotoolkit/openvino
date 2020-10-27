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
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_single_indices)
{
    Shape params_shape{3, 3};
    Shape indices_shape{2};
    Shape out_shape{};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v5::GatherND>(P, I);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1, 2});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.5f}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    auto f5 = make_shared<Function>(G5, ParameterVector{P, I});
    auto c5 = backend->compile(f5);
    c5->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.5f}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_scalar_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v5::GatherND>(P, I);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0f, 1.1f, 1.2f, 1.3f});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 0, 1, 1});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.0f, 1.3f}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    auto f5 = make_shared<Function>(G5, ParameterVector{P, I});
    auto c5 = backend->compile(f5);
    c5->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.0f, 1.3f}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_1d_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 1};
    Shape out_shape{2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v5::GatherND>(P, I);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0f, 1.1f, 1.2f, 1.3f});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1, 0});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{1.2f, 1.3f, 1.0f, 1.1f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    auto f5 = make_shared<Function>(G5, ParameterVector{P, I});
    auto c5 = backend->compile(f5);
    c5->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{1.2f, 1.3f, 1.0f, 1.1f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_scalar_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 3};
    Shape out_shape{2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v5::GatherND>(P, I);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f, 2.2f, 2.3f});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 0, 1, 1, 0, 1});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.1f, 2.1f}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    auto f5 = make_shared<Function>(G5, ParameterVector{P, I});
    auto c5 = backend->compile(f5);
    c5->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.1f, 2.1f}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_1d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v5::GatherND>(P, I);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f, 2.2f, 2.3f});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 0});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{1.2f, 1.3f, 2.0f, 2.1f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    auto f5 = make_shared<Function>(G5, ParameterVector{P, I});
    auto c5 = backend->compile(f5);
    c5->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{1.2f, 1.3f, 2.0f, 2.1f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_2d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{1, 1};
    Shape out_shape{1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v5::GatherND>(P, I);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f, 2.2f, 2.3f});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{2.0f, 2.1f, 2.2f, 2.3f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    auto f5 = make_shared<Function>(G5, ParameterVector{P, I});
    auto c5 = backend->compile(f5);
    c5->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{2.0f, 2.1f, 2.2f, 2.3f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_scalar_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 1, 2};
    Shape out_shape{2, 1};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v5::GatherND>(P, I);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0f, 1.1f, 1.2f, 1.3f});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 0, 0, 1});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.0f, 1.1f}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    auto f5 = make_shared<Function>(G5, ParameterVector{P, I});
    auto c5 = backend->compile(f5);
    c5->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.0f, 1.1f}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_1d_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v5::GatherND>(P, I);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0f, 1.1f, 1.2f, 1.3f});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1, 0});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{1.2f, 1.3f, 1.0f, 1.1f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    auto f5 = make_shared<Function>(G5, ParameterVector{P, I});
    auto c5 = backend->compile(f5);
    c5->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{1.2f, 1.3f, 1.0f, 1.1f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_scalar_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 2, 3};
    Shape out_shape{2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v5::GatherND>(P, I);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f, 2.2f, 2.3f});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{1.1f, 2.1f, 1.3f, 2.2f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    auto f5 = make_shared<Function>(G5, ParameterVector{P, I});
    auto c5 = backend->compile(f5);
    c5->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{1.1f, 2.1f, 1.3f, 2.2f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_1d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v5::GatherND>(P, I);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f, 2.2f, 2.3f});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 0, 0, 0, 1, 1});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{1.2f, 1.3f, 2.0f, 2.1f, 1.0f, 1.1f, 2.2f, 2.3f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    auto f5 = make_shared<Function>(G5, ParameterVector{P, I});
    auto c5 = backend->compile(f5);
    c5->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{1.2f, 1.3f, 2.0f, 2.1f, 1.0f, 1.1f, 2.2f, 2.3f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_2d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v5::GatherND>(P, I);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f, 2.2f, 2.3f});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1, 0});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{2.0f, 2.1f, 2.2f, 2.3f, 1.0f, 1.1f, 1.2f, 1.3f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));

    auto G5 = make_shared<op::v5::GatherND>(P, I);
    auto f5 = make_shared<Function>(G5, ParameterVector{P, I});
    auto c5 = backend->compile(f5);
    c5->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{2.0f, 2.1f, 2.2f, 2.3f, 1.0f, 1.1f, 1.2f, 1.3f}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_dims1)
{
    Shape params_shape{2, 3, 4};
    Shape indices_shape{2, 1};
    Shape out_shape{2, 4};
    int batch_dims = 1;
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v5::GatherND>(P, I, batch_dims);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1, 0});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{5, 6, 7, 8, 13, 14, 15, 16}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_dims2)
{
    Shape params_shape{2, 3, 4, 2};
    Shape indices_shape{2, 3, 3, 2};
    Shape out_shape{6, 3};
    int batch_dims = 2;
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v5::GatherND>(P, I, batch_dims);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                               17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                               33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1, 0, 3, 1, 2, 1, 0, 1, 1, 1, 2, 0, 3, 0, 3, 1, 2, 1,
                                 2, 0, 1, 1, 3, 1, 1, 1, 2, 0, 2, 0, 0, 0, 3, 1, 3, 1});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{3, 8, 6, 10, 12, 13, 23, 24, 22, 29, 28, 32, 36, 37, 37, 41, 48, 48}),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_dims2_lead_dims)
{
    Shape params_shape{2, 3, 4};
    Shape indices_shape{2, 3, 1, 1};
    Shape out_shape{6, 1};
    int batch_dims = 2;
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::v5::GatherND>(P, I, batch_dims);
    auto f = make_shared<Function>(G, ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1, 0, 2, 0, 2, 2});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 5, 11, 13, 19, 23}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}
