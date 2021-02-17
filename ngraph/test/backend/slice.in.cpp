//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include "ngraph/builder/reshape.hpp"
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

NGRAPH_TEST(${BACKEND_NAME}, slice_scalar)
{
    Shape shape_a{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{};
    auto r = make_shared<op::Slice>(A, Coordinate{}, Coordinate{});
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(r), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{312});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{312}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{3, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 1}, Coordinate{3, 3});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{2, 3, 6, 7, 10, 11}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_vector)
{
    Shape shape_a{16};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{12};
    auto r = make_shared<op::Slice>(A, Coordinate{2}, Coordinate{14});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_axis_0_overlap)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto C = make_shared<op::v1::Add>(A, B);
    Shape shape_r{2, 4};
    auto D = make_shared<op::Slice>(C, Coordinate{0, 0}, Coordinate{2, 4});
    auto E = make_shared<op::Slice>(C, Coordinate{1, 0}, Coordinate{3, 4});
    auto r = make_shared<op::v1::Add>(D, E);
    auto f = make_shared<Function>(r, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, shape_a);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{12, 16, 20, 24, 28, 32, 36, 40}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_axis_0_in_place)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 4};
    auto D = make_shared<op::Slice>(A, Coordinate{0, 0}, Coordinate{2, 4});
    auto E = make_shared<op::Slice>(A, Coordinate{2, 0}, Coordinate{4, 4});
    auto r = make_shared<op::v1::Add>(D, E);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{10, 12, 14, 16, 18, 20, 22, 24}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_axis_0_in_place_twice)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{1, 4};
    auto B = make_shared<op::Slice>(A, Coordinate{0, 0}, Coordinate{2, 4});
    auto D = make_shared<op::Slice>(B, Coordinate{1, 0}, Coordinate{2, 4});
    auto E = make_shared<op::Slice>(A, Coordinate{2, 0}, Coordinate{3, 4});
    auto r = make_shared<op::v1::Add>(D, E);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{14, 16, 18, 20}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_axis_0_in_place_twice_overlap)
{
    Shape shape_a{5, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 4};
    auto B = make_shared<op::Slice>(A, Coordinate{1, 0}, Coordinate{5, 4});
    auto D = make_shared<op::Slice>(B, Coordinate{1, 0}, Coordinate{3, 4});
    auto E = make_shared<op::Slice>(B, Coordinate{2, 0}, Coordinate{4, 4});
    auto r = make_shared<op::v1::Add>(D, E);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{22, 24, 26, 28, 30, 32, 34, 36}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_axis_0_in_place_with_transpose)
{
    Shape shape_a{4, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 4};
    auto B = make_shared<op::Slice>(A, Coordinate{1, 0}, Coordinate{4, 5});
    auto C = builder::opset1::transpose(B);
    auto D = make_shared<op::Slice>(C, Coordinate{1, 0}, Coordinate{5, 3});
    auto E = builder::opset1::transpose(D);
    auto r = make_shared<op::Slice>(E, Coordinate{1, 0}, Coordinate{3, 4});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{12, 13, 14, 15, 17, 18, 19, 20}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_strided)
{
    Shape shape_a{4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{1, 0}, Coordinate{4, 4}, Strides{2, 3});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{4, 7, 12, 15}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{1, 1, 1}, Coordinate{3, 3, 3});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{21, 22, 25, 26, 37, 38, 41, 42}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_strided)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 2});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0, 2, 8, 10, 32, 34, 40, 42}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_strided_different_strides)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 3});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0, 3, 8, 11, 32, 35, 40, 43}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_strided_different_strides_int64)
{
    Shape shape_a{4, 4, 4};
    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    Shape shape_r{2, 2, 2};
    auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 3});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape_a);
    copy_data(a, vector<int64_t>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                                 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                                 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->create_tensor(element::i64, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int64_t>{0, 3, 8, 11, 32, 35, 40, 43}), read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_start_just_oob)
{
    Shape shape_a{20, 10, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_r{20, 0, 5};
    auto r =
        make_shared<op::Slice>(A, Coordinate{0, 10, 0}, Coordinate{20, 10, 5}, Strides{1, 1, 1});
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    vector<float> a_data(20 * 10 * 5, 222.0f);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::f32, shape_r);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{}), read_vector<float>(result)));
}
