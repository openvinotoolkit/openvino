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

// Trivial case.
NGRAPH_TEST(${BACKEND_NAME}, argmin_trivial)
{
    Shape shape{4, 3};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMin>(A, 0, element::i32), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result = backend->create_tensor(element::i32, rshape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int>{3, 2, 1}), read_vector<int>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmin_2D_i32)
{
    Shape shape{4, 3};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMin>(A, 0, element::i32), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result = backend->create_tensor(element::i32, rshape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int>{3, 2, 1}), read_vector<int>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmin_3D_i32)
{
    Shape shape{3, 3, 4};
    Shape rshape{3, 4};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMin>(A, 1, element::i32), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a,
              test::NDArray<int, 3>({{{12, 2, 10, 9}, {3, 5, 0, 8}, {7, 9, 1, 5}},
                                     {{7, 2, 4, 10}, {6, 10, 2, 2}, {12, 1, 1, 1}},
                                     {{10, 2, 2, 4}, {1, 5, 5, 1}, {7, 12, 2, 2}}})
                  .get_vector());
    auto result = backend->create_tensor(element::i32, rshape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int>{1, 0, 1, 2, 1, 2, 2, 2, 1, 0, 0, 1}), read_vector<int>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmin_3D_i64)
{
    Shape shape{3, 3, 4};
    Shape rshape{3, 4};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMin>(A, 1, element::i64), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a,
              test::NDArray<int, 3>({{{12, 2, 10, 9}, {3, 5, 0, 8}, {7, 9, 1, 5}},
                                     {{7, 2, 4, 10}, {6, 10, 2, 2}, {12, 1, 1, 1}},
                                     {{10, 2, 2, 4}, {1, 5, 5, 1}, {7, 12, 2, 2}}})
                  .get_vector());
    auto result = backend->create_tensor(element::i64, rshape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int64_t>{1, 0, 1, 2, 1, 2, 2, 2, 1, 0, 0, 1}), read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmin_4D_i64)
{
    Shape shape{2, 2, 5, 5}; // NCHW ->(0,1,2,3)
    Shape rshape{2, 2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMin>(A, 3, element::i64), ParameterVector{A});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(
        a,
        test::NDArray<int, 4>(
            {{{{3, 1, 1, 2, 105},
               {0, 3, 2, 1, 2},
               {2, 4, 2, 0, 1},
               {2, 5, 1, 1, 22},
               {5, 2, 1, 7, 5}},
              {{3, 1, 2, 2, 1},
               {1, 7, 3, 8, 1},
               {2, 10, 1, 3, 2},
               {3, 1, 0, 0, 6},
               {2, 0, 0, 0, 0}}},
             {{{0, 2, 1, 1, 0}, {0, 0, 0, 0, 1}, {0, 0, 1, 0, 3}, {2, 0, 0, 3, 0}, {0, 0, 0, 0, 1}},
              {{2, 1, 0, 0, 1},
               {0, 2, 0, 0, 0},
               {1, 1, 2, 0, 2},
               {1, 1, 1, 0, 1},
               {1, 0, 0, 0, 2}}}})
            .get_vector());
    auto result = backend->create_tensor(element::i64, rshape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int64_t>{1, 0, 3, 2, 2, 1, 0, 2, 2, 1, 0, 0, 0, 1, 0, 2, 0, 3, 3, 1}),
              read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmin_4D_axis_3_i64)
{
    Shape shape{2, 2, 5, 5}; // NCHW ->(0,1,2,3)
    Shape rshape{2, 2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMin>(A, 3, element::i64), ParameterVector{A});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 4>({{{{0.5f, 1.5f, 0.8f, 2.9f, 1.05f}, // img 0 ch 0
                                         {0.5f, 3.5f, 2.0f, 1.0f, 0.2f},
                                         {2.0f, 0.0f, 2.2f, 0.2f, 1.4f},
                                         {2.9f, 0.0f, 1.52f, 1.2f, 2.22f},
                                         {5.0f, 2.0f, 1.0f, 0.5f, 0.85f}},
                                        {{0.25f, 0.02f, 0.02f, 2.2f, 0.001f}, // img 0 ch 1
                                         {1.0f, 0.2f, 3.0f, 0.25f, 1.14f},
                                         {2.25f, 10.1f, 1.0f, 0.02f, 2.22f},
                                         {3.2f, 1.002f, 0.001f, 0.2f, 6.0f},
                                         {2.0f, 0.0f, 0.0f, 0.0f, 0.0f}}},
                                       {{{0.0f, 2.2f, 1.2f, 1.6f, 0.2f}, // img 1 ch 0
                                         {0.01f, 0.0f, 0.22f, 0.02f, 1.1f},
                                         {0.01f, 0.5f, 1.6f, 0.2f, 3.2f},
                                         {2.4f, 0.5f, 0.0f, 3.0f, 0.1f},
                                         {0.0f, 0.5f, 0.4f, 0.8f, 1.0f}},
                                        {{2.0f, 1.0f, 0.0f, 0.0f, 1.0f}, // img 1 ch 1
                                         {0.0f, 2.0f, 0.0f, 0.0f, 0.0f},
                                         {1.0f, 1.0f, 2.0f, 0.0f, 2.0f},
                                         {1.0f, 1.0f, 1.0f, 0.0f, 1.0f},
                                         {1.0f, 0.0f, 0.0f, 0.0f, 2.0f}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::i64, rshape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((test::NDArray<int64_t, 3>({{{0, 4, 1, 1, 3},   // ch0
                                           {4, 1, 3, 2, 1}},  //
                                          {{0, 1, 0, 2, 0},   // ch1
                                           {2, 0, 3, 3, 1}}}) //
                   .get_vector()),
              read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmin_4D_axis_3)
{
    Shape shape{2, 2, 5, 5}; // NCHW ->(0,1,2,3)
    Shape rshape{2, 2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMin>(A, 3, element::i32), ParameterVector{A});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 4>({{{{0.5f, 1.5f, 0.8f, 2.9f, 1.05f}, // img 0 ch 0
                                         {0.5f, 3.5f, 2.0f, 1.0f, 0.2f},
                                         {2.0f, 0.0f, 2.2f, 0.2f, 1.4f},
                                         {2.9f, 0.0f, 1.52f, 1.2f, 2.22f},
                                         {5.0f, 2.0f, 1.0f, 0.5f, 0.85f}},
                                        {{0.25f, 0.02f, 0.02f, 2.2f, 0.001f}, // img 0 ch 1
                                         {1.0f, 0.2f, 3.0f, 0.25f, 1.14f},
                                         {2.25f, 10.1f, 1.0f, 0.02f, 2.22f},
                                         {3.2f, 1.002f, 0.001f, 0.2f, 6.0f},
                                         {2.0f, 0.0f, 0.0f, 0.0f, 0.0f}}},
                                       {{{0.0f, 2.2f, 1.2f, 1.6f, 0.2f}, // img 1 ch 0
                                         {0.01f, 0.0f, 0.22f, 0.02f, 1.1f},
                                         {0.01f, 0.5f, 1.6f, 0.2f, 3.2f},
                                         {2.4f, 0.5f, 0.0f, 3.0f, 0.1f},
                                         {0.0f, 0.5f, 0.4f, 0.8f, 1.0f}},
                                        {{2.0f, 1.0f, 0.0f, 0.0f, 1.0f}, // img 1 ch 1
                                         {0.0f, 2.0f, 0.0f, 0.0f, 0.0f},
                                         {1.0f, 1.0f, 2.0f, 0.0f, 2.0f},
                                         {1.0f, 1.0f, 1.0f, 0.0f, 1.0f},
                                         {1.0f, 0.0f, 0.0f, 0.0f, 2.0f}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::i32, rshape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((test::NDArray<int, 3>({{{0, 4, 1, 1, 3},   // ch0
                                       {4, 1, 3, 2, 1}},  //
                                      {{0, 1, 0, 2, 0},   // ch1
                                       {2, 0, 3, 3, 1}}}) //
                   .get_vector()),
              read_vector<int>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmax_trivial)
{
    Shape shape{4, 3}; // HW -> (0,1)
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMax>(A, 0, element::i32), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{9, 2, 10, 12, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result = backend->create_tensor(element::i32, rshape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int>{1, 3, 0}), read_vector<int>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmax_2D_i32)
{
    Shape shape{4, 3};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMax>(A, 0, element::i32), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result = backend->create_tensor(element::i32, rshape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int>{0, 3, 0}), read_vector<int>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmax_3D_i32)
{
    Shape shape{3, 3, 4};
    Shape rshape{3, 4};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMax>(A, 1, element::i32), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a,
              test::NDArray<int, 3>({{{12, 2, 10, 9}, {3, 5, 0, 8}, {7, 9, 1, 5}},
                                     {{7, 2, 4, 10}, {6, 10, 2, 2}, {12, 1, 1, 1}},
                                     {{10, 2, 2, 4}, {1, 5, 5, 1}, {7, 12, 2, 2}}})
                  .get_vector());
    auto result = backend->create_tensor(element::i32, rshape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int>{0, 2, 0, 0, 2, 1, 0, 0, 0, 2, 1, 0}), read_vector<int>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmax_3D_i64)
{
    Shape shape{3, 3, 4};
    Shape rshape{3, 4};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMax>(A, 1, element::i64), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a,
              test::NDArray<int, 3>({{{12, 2, 10, 9}, {3, 5, 0, 8}, {7, 9, 1, 5}},
                                     {{7, 2, 4, 10}, {6, 10, 2, 2}, {12, 1, 1, 1}},
                                     {{10, 2, 2, 4}, {1, 5, 5, 1}, {7, 12, 2, 2}}})
                  .get_vector());
    auto result = backend->create_tensor(element::i64, rshape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int64_t>{0, 2, 0, 0, 2, 1, 0, 0, 0, 2, 1, 0}), read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmax_4D_i64)
{
    Shape shape{2, 2, 5, 5}; // NCHW ->(0,1,2,3)
    Shape rshape{2, 2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMax>(A, 3, element::i64), ParameterVector{A});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(
        a,
        test::NDArray<int, 4>(
            {{{{3, 1, 1, 2, 105},
               {0, 3, 2, 1, 2},
               {2, 4, 2, 0, 1},
               {2, 5, 1, 1, 22},
               {5, 2, 1, 7, 5}},
              {{3, 1, 2, 2, 1},
               {1, 7, 3, 8, 1},
               {2, 10, 1, 3, 2},
               {3, 1, 0, 0, 6},
               {2, 0, 0, 0, 0}}},
             {{{0, 2, 1, 1, 0}, {0, 0, 0, 0, 1}, {0, 0, 1, 0, 3}, {2, 0, 0, 3, 0}, {0, 0, 0, 0, 1}},
              {{2, 1, 0, 0, 1},
               {0, 2, 0, 0, 0},
               {1, 1, 2, 0, 2},
               {1, 1, 1, 0, 1},
               {1, 0, 0, 0, 2}}}})
            .get_vector());
    auto result = backend->create_tensor(element::i64, rshape);
    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int64_t>{4, 1, 1, 4, 3, 0, 3, 1, 4, 0, 1, 4, 4, 3, 4, 0, 1, 2, 0, 4}),
              read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmax_3D_axis_0) // Along Channels
{
    Shape shape{3, 4, 2}; // CHW ->(0,1,2)
    Shape rshape{4, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMax>(A, 0, element::i32), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);

    copy_data(a,
              test::NDArray<float, 3>({{{8, 4}, // ch0
                                        {12, 10},
                                        {2, 9},
                                        {1, 5}},

                                       {{6, 7}, // ch1
                                        {11, 3},
                                        {9, 2},
                                        {10, 12}},

                                       {{8, 4}, // ch2
                                        {6, 1},
                                        {5, 3},
                                        {11, 7}}})
                  .get_vector());
    auto result = backend->create_tensor(element::i32, rshape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((test::NDArray<int, 2>({{0, 1},  // r0
                                      {0, 0},  // r1
                                      {1, 0},  // r2
                                      {2, 1}}) // r3
                   .get_vector()),
              read_vector<int>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmax_3D_axis_1) // Along Height
{
    Shape shape{3, 4, 2}; // CHW ->(0,1,2)
    Shape rshape{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMax>(A, 1, element::i32), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{8, 4}, // ch0
                                        {12, 10},
                                        {2, 9},
                                        {1, 5}},

                                       {{6, 7}, // ch1
                                        {11, 3},
                                        {9, 2},
                                        {10, 12}},

                                       {{8, 4}, // ch2
                                        {6, 1},
                                        {5, 3},
                                        {11, 7}}})
                  .get_vector());
    auto result = backend->create_tensor(element::i32, rshape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((test::NDArray<int, 2>({{1, 1}, //
                                      {1, 3}, //
                                      {3, 3}})
                   .get_vector()),
              read_vector<int>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmax_3D_axis_2) // Along Width
{
    Shape shape{3, 4, 2}; // CHW ->(0,1,2)
    Shape rshape{3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMax>(A, 2, element::i32), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 3>({{{8, 4}, // ch0
                                        {12, 10},
                                        {2, 9},
                                        {1, 5}},

                                       {{6, 7}, // ch1
                                        {11, 3},
                                        {9, 2},
                                        {10, 12}},

                                       {{8, 4}, // ch2
                                        {6, 1},
                                        {5, 3},
                                        {11, 7}}})
                  .get_vector());
    auto result = backend->create_tensor(element::i32, rshape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((test::NDArray<int, 2>({{0, 0, 1, 1},  //
                                      {1, 0, 0, 1},  //
                                      {0, 0, 0, 0}}) //
                   .get_vector()),
              read_vector<int>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmax_4D_axis_3)
{
    Shape shape{2, 2, 5, 5}; // NCHW ->(0,1,2,3)
    Shape rshape{2, 2, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMax>(A, 3, element::i32), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1}, // img 0 ch 0
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}},

                                        {{0, 0, 0, 2, 0}, // img 0 ch 1
                                         {0, 2, 3, 0, 1},
                                         {2, 0, 1, 0, 2},
                                         {3, 1, 0, 0, 0},
                                         {2, 0, 0, 0, 0}}},

                                       {{{0, 2, 1, 1, 0}, // img 1 ch 0
                                         {0, 0, 2, 0, 1},
                                         {0, 0, 1, 2, 3},
                                         {2, 0, 0, 3, 0},
                                         {0, 0, 0, 0, 0}},

                                        {{2, 1, 0, 0, 1}, // img 1 ch 1
                                         {0, 2, 0, 0, 0},
                                         {1, 1, 2, 0, 2},
                                         {1, 1, 1, 0, 1},
                                         {1, 0, 0, 0, 2}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::i32, rshape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((test::NDArray<int, 3>({{{3, 1, 0, 0, 1}, {3, 2, 0, 0, 0}},  // ch0
                                      {{1, 2, 4, 3, 0}, {0, 1, 2, 0, 4}}}) // ch1
                   .get_vector()),
              read_vector<int>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmin_trivial_in_i32)
{
    Shape shape{4, 3};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMin>(A, 0, element::i32), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result = backend->create_tensor(element::i32, rshape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int>{3, 2, 1}), read_vector<int>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmax_4D_axis_3_i64_in_i32)
{
    Shape shape{2, 2, 5, 5}; // NCHW ->(0,1,2,3)
    Shape rshape{2, 2, 5};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMax>(A, 3, element::i64), ParameterVector{A});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a,
              test::NDArray<int32_t, 4>({{{{0, 1, 0, 2, 1}, // img 0 ch 0
                                           {0, 3, 2, 0, 0},
                                           {2, 0, 0, 0, 1},
                                           {2, 0, 1, 1, 2},
                                           {0, 2, 1, 0, 0}},

                                          {{0, 0, 0, 2, 0}, // img 0 ch 1
                                           {0, 2, 3, 0, 1},
                                           {2, 0, 1, 0, 2},
                                           {3, 1, 0, 0, 0},
                                           {2, 0, 0, 0, 0}}},

                                         {{{0, 2, 1, 1, 0}, // img 1 ch 0
                                           {0, 0, 2, 0, 1},
                                           {0, 0, 1, 2, 3},
                                           {2, 0, 0, 3, 0},
                                           {0, 0, 0, 0, 0}},

                                          {{2, 1, 0, 0, 1}, // img 1 ch 1
                                           {0, 2, 0, 0, 0},
                                           {1, 1, 2, 0, 2},
                                           {1, 1, 1, 0, 1},
                                           {1, 0, 0, 0, 2}}}})
                  .get_vector());
    auto result = backend->create_tensor(element::i64, rshape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((test::NDArray<int64_t, 3>({{{3, 1, 0, 0, 1}, {3, 2, 0, 0, 0}},  // ch0
                                          {{1, 2, 4, 3, 0}, {0, 1, 2, 0, 4}}}) // ch1
                   .get_vector()),
              read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, argmin_trivial_in_double)
{
    Shape shape{4, 3};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::f64, shape);
    auto f = make_shared<Function>(make_shared<op::ArgMin>(A, 0, element::i32), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f64, shape);
    copy_data(a, vector<double>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result = backend->create_tensor(element::i32, rshape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{3, 2, 1}), read_vector<int32_t>(result));
}
