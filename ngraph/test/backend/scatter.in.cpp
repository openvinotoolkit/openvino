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
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

#if 0
NGRAPH_TEST(${BACKEND_NAME}, scatter_add_4d_indices)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{2, 3, 4, 2};
    Shape updates_shape{2, 3, 4, 2, 3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    auto G = make_shared<op::ScatterAdd>(R, I, U);
    auto f =
        make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{R, I, U});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto r = backend->create_tensor(element::f32, ref_shape);
    copy_data(r, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5,
                               6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
                                 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
                                 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2});
    auto u = backend->create_tensor(element::f32, updates_shape);
    copy_data(u,
              vector<float>{
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {r, i, u});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{0,   17,  34,  51,  68, 85, 102, 119, 136, 17, 34,  51,  68, 85,
                       102, 119, 136, 153, 0,  17, 34,  51,  68,  85, 102, 119, 136}),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}
#endif

NGRAPH_TEST(${BACKEND_NAME}, scatter_add_3d_indices)
{
    Shape ref_shape{2, 3, 3};
    Shape indices_shape{2, 2, 2};
    Shape updates_shape{2, 2, 2, 3, 3};
    Shape out_shape{2, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    auto G = make_shared<op::ScatterAdd>(R, I, U);
    auto f =
        make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{R, I, U});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto r = backend->create_tensor(element::f32, ref_shape);
    copy_data(r, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 0, 0, 1, 1, 0});
    auto u = backend->create_tensor(element::f32, updates_shape);
    copy_data(u, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                               0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {r, i, u});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{0, 5, 10, 15, 20, 25, 30, 35, 40, 5, 10, 15, 20, 25, 30, 35, 40, 45}),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, scatter_add_2d_indices)
{
    Shape ref_shape{3};
    Shape indices_shape{2, 2};
    Shape updates_shape{2, 2};
    Shape out_shape{3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    auto G = make_shared<op::ScatterAdd>(R, I, U);
    auto f =
        make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{R, I, U});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto r = backend->create_tensor(element::f32, ref_shape);
    copy_data(r, vector<float>{0, 1, 2});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 0});
    auto u = backend->create_tensor(element::f32, updates_shape);
    copy_data(u, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {r, i, u});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{5, 6, 2}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, scatter_add_1d_indices)
{
    Shape ref_shape{2, 3, 3};
    Shape indices_shape{2};
    Shape updates_shape{2, 3, 3};
    Shape out_shape{2, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    auto G = make_shared<op::ScatterAdd>(R, I, U);
    auto f =
        make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{R, I, U});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto r = backend->create_tensor(element::f32, ref_shape);
    copy_data(r, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1, 0});
    auto u = backend->create_tensor(element::f32, updates_shape);
    copy_data(u, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {r, i, u});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{0, 2, 4, 6, 8, 10, 12, 14, 16, 2, 4, 6, 8, 10, 12, 14, 16, 18}),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, scatter_add_scalar_indices)
{
    Shape ref_shape{2, 3, 3};
    Shape indices_shape{};
    Shape updates_shape{3, 3};
    Shape out_shape{2, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    auto G = make_shared<op::ScatterAdd>(R, I, U);
    auto f =
        make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{R, I, U});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto r = backend->create_tensor(element::f32, ref_shape);
    copy_data(r, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1});
    auto u = backend->create_tensor(element::f32, updates_shape);
    copy_data(u, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {r, i, u});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8, 10, 12, 14, 16, 18}),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, scatter_nd_add_batch_2d_to_3d)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{2, 1};
    Shape updates_shape{2, 3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    auto G = make_shared<op::ScatterNDAdd>(R, I, U);
    auto f =
        make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{R, I, U});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto r = backend->create_tensor(element::f32, ref_shape);
    copy_data(r, vector<float>{1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5,
                               5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 2});
    auto u = backend->create_tensor(element::f32, updates_shape);
    copy_data(u, vector<float>{1, 1, 1, 2, 2, 2, 3, 3, 3, 7, 7, 7, 8, 8, 8, 9, 9, 9});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {r, i, u});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 2, 2, 4, 4,  4,  6,  6,  6,  4,  4,  4,  5, 5,
                                                 5, 6, 6, 6, 14, 14, 14, 16, 16, 16, 18, 18, 18}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, scatter_nd_add_2d_to_3d)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    auto G = make_shared<op::ScatterNDAdd>(R, I, U);
    auto f =
        make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{R, I, U});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto r = backend->create_tensor(element::f32, ref_shape);
    copy_data(r, vector<float>{1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5,
                               5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0});
    auto u = backend->create_tensor(element::f32, updates_shape);
    copy_data(u, vector<float>{1, 1, 1, 2, 2, 2, 3, 3, 3});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {r, i, u});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 2, 2, 4, 4, 4, 6, 6, 6, 4, 4, 4, 5, 5,
                                                 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}
