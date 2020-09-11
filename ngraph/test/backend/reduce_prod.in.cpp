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
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

// ----------------------- keep dims = false ----------------------- //

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, Shape{});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{24}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_matrix_columns)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{15, 48}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_matrix_rows)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 12, 30}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_matrix_rows_zero)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 1, 1}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 1}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_vector_zero)
{
    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_matrix_to_scalar_zero_by_zero)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_3d_to_matrix_most_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1 * 10 * 19,
                                                 2 * 11 * 20,
                                                 3 * 12 * 21,
                                                 4 * 13 * 22,
                                                 5 * 14 * 23,
                                                 6 * 15 * 24,
                                                 7 * 16 * 25,
                                                 8 * 17 * 26,
                                                 9 * 18 * 27}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_3d_to_matrix_least_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 2);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1 * 2 * 3,
                                                 4 * 5 * 6,
                                                 7 * 8 * 9,
                                                 10 * 11 * 12,
                                                 13 * 14 * 15,
                                                 16 * 17 * 18,
                                                 19 * 20 * 21,
                                                 22 * 23 * 24,
                                                 25 * 26 * 27}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.0f * 10.0f * 19.0f * 4.0f * 13.0f * 22.0f * 7.0f * 16.0f * 25.0f,
                       2.0f * 11.0f * 20.0f * 5.0f * 14.0f * 23.0f * 8.0f * 17.0f * 26.0f,
                       3.0f * 12.0f * 21.0f * 6.0f * 15.0f * 24.0f * 9.0f * 18.0f * 27.0f}),
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_3d_to_scalar)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto axes = make_shared<op::Constant>(element::i32, Shape{3}, vector<int32_t>{0, 1, 2});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{1.0f * 10.0f * 9.0f * 4.0f * 13.0f * 6.0f * 7.0f *
                                                12.0f * 3.0f * 2.0f * 11.0f * 8.0f * 5.0f * 14.0f *
                                                5.0f * 8.0f * 11.0f * 2.0f * 3.0f * 12.0f * 7.0f *
                                                6.0f * 13.0f * 4.0f * 9.0f * 10.0f * 1.0f},
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_3d_eliminate_zero_dim)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the
    // right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 1, 1, 1, 1, 1}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_2d_to_scalar_int32)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ(vector<int32_t>{1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9}, read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_to_scalar_int32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::i32, Shape{});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{24}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_to_scalar_int8)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i8, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape);
    copy_data(a, vector<int8_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::i8, Shape{});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int8_t>{24}), read_vector<int8_t>(result));
}

// ----------------------- keep dims = true ----------------------- //

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, Shape{1, 1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{24}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_matrix_columns)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{15, 48}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_matrix_rows)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 12, 30}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_matrix_rows_zero)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 1, 1}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 1}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_vector_zero)
{
    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_matrix_to_scalar_zero_by_zero)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_3d_to_matrix_most_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 3, 3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1 * 10 * 19,
                                                 2 * 11 * 20,
                                                 3 * 12 * 21,
                                                 4 * 13 * 22,
                                                 5 * 14 * 23,
                                                 6 * 15 * 24,
                                                 7 * 16 * 25,
                                                 8 * 17 * 26,
                                                 9 * 18 * 27}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_3d_to_matrix_least_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 2);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1 * 2 * 3,
                                                 4 * 5 * 6,
                                                 7 * 8 * 9,
                                                 10 * 11 * 12,
                                                 13 * 14 * 15,
                                                 16 * 17 * 18,
                                                 19 * 20 * 21,
                                                 22 * 23 * 24,
                                                 25 * 26 * 27}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 1, 3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.0f * 10.0f * 19.0f * 4.0f * 13.0f * 22.0f * 7.0f * 16.0f * 25.0f,
                       2.0f * 11.0f * 20.0f * 5.0f * 14.0f * 23.0f * 8.0f * 17.0f * 26.0f,
                       3.0f * 12.0f * 21.0f * 6.0f * 15.0f * 24.0f * 9.0f * 18.0f * 27.0f}),
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_3d_to_scalar)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 1, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{3}, vector<int32_t>{0, 1, 2});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{1.0f * 10.0f * 9.0f * 4.0f * 13.0f * 6.0f * 7.0f *
                                                12.0f * 3.0f * 2.0f * 11.0f * 8.0f * 5.0f * 14.0f *
                                                5.0f * 8.0f * 11.0f * 2.0f * 3.0f * 12.0f * 7.0f *
                                                6.0f * 13.0f * 4.0f * 9.0f * 10.0f * 1.0f},
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_3d_eliminate_zero_dim)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 1, 2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the
    // right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 1, 1, 1, 1, 1}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_2d_to_scalar_int32)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{1, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ(vector<int32_t>{1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9}, read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_to_scalar_int32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::i32, Shape{1, 1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{24}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_to_scalar_int8)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i8, shape);
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape);
    copy_data(a, vector<int8_t>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::i8, Shape{1, 1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int8_t>{24}), read_vector<int8_t>(result));
}

// Dynamic

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_matrix_columns_dynamic)
{
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    Shape shape_a{3, 2};
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{15, 48}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_matrix_rows_dynamic)
{
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    Shape shape_a{3, 2};
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 12, 30}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_matrix_columns_dynamic)
{
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    Shape shape_a{3, 2};
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{15, 48}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_product_keep_matrix_rows_dynamic)
{
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceProd>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}", true);

    // Create some tensors for input/output
    Shape shape_a{3, 2};
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 12, 30}), read_vector<float>(result)));
}
