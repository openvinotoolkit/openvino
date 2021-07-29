// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/test_control.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"


using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

TEST(op_eval, reduce_sum_matrix_rows_zero)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0, 0, 0}), read_vector<float>(result)));
}

TEST(op_eval, reduce_sum_vector_zero)
{
    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0}), read_vector<float>(result)));
}


TEST(op_eval, reduce_sum_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0, 0}), read_vector<float>(result)));
}

TEST(op_eval, reduce_sum_matrix_to_scalar_zero_by_zero)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0}), read_vector<float>(result)));
}

TEST(op_eval, reduce_sum_3d_eliminate_zero_dim)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the
    // right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0, 0, 0, 0, 0, 0}), read_vector<float>(result)));
}

TEST(op_eval, reduce_sum_3d_eliminate_zero_dim_int32)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{3, 2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, false), ParameterVector{A});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{});
    auto result = backend->create_tensor(element::i32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the
    // right value.
    copy_data(result, vector<int32_t>{2112, 2112, 2112, 2112, 2112, 2112});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{0, 0, 0, 0, 0, 0}), read_vector<int32_t>(result));
}

TEST(op_eval, reduce_sum_dynamic)
{
    // Create a graph for f(x,axes:int32) = Sum(x,Convert<int64>(axes)).
    auto x = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto axes_i64 = make_shared<op::Convert>(axes, element::i64);

    auto sum = make_shared<op::v1::ReduceSum>(x, axes_i64, false);
    ASSERT_TRUE(sum->get_output_partial_shape(0).rank().is_dynamic());

    auto f = make_shared<Function>(NodeVector{sum}, ParameterVector{x, axes});

    auto backend = runtime::Backend::create("INTERPRETER", true);

    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    std::vector<Shape> x_shapes{
        Shape{2, 3}, Shape{2, 3}, Shape{2, 3}, Shape{2, 3}, Shape{5}, Shape{5}};
    std::vector<std::vector<int32_t>> axeses{{}, {0}, {1}, {0, 1}, {}, {0}};
    std::vector<std::vector<float>> inputs{{1, 2, 3, 4, 5, 6},
                                           {1, 2, 3, 4, 5, 6},
                                           {1, 2, 3, 4, 5, 6},
                                           {1, 2, 3, 4, 5, 6},
                                           {1, 2, 3, 4, 5},
                                           {1, 2, 3, 4, 5}};
    std::vector<Shape> expected_result_shapes{
        Shape{2, 3}, Shape{3}, Shape{2}, Shape{}, Shape{5}, Shape{}};
    std::vector<std::vector<float>> expected_results{
        {1, 2, 3, 4, 5, 6}, {5, 7, 9}, {6, 15}, {21}, {1, 2, 3, 4, 5}, {15}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::f32, x_shapes[i]);
        auto t_axes = backend->create_tensor(element::i32, Shape{axeses[i].size()});

        copy_data(t_x, inputs[i]);
        copy_data(t_axes, axeses[i]);

        ex->call_with_validate({t_r}, {t_x, t_axes});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);

        auto results = read_vector<float>(t_r);

        ASSERT_TRUE(test::all_close_f(results, expected_results[i], MIN_FLOAT_TOLERANCE_BITS));
    }
}

TEST(op_eval, reduce_sum_keep_matrix_rows_zero)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0, 0, 0}), read_vector<float>(result)));
}

TEST(op_eval, reduce_sum_keep_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0, 0}), read_vector<float>(result)));
}

TEST(op_eval, reduce_sum_keep_vector_zero)
{
    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 0);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0}), read_vector<float>(result)));
}

TEST(op_eval, reduce_sum_keep_matrix_to_scalar_zero_by_zero)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{1, 1};
    auto axes = make_shared<op::Constant>(element::i32, Shape{2}, vector<int32_t>{0, 1});
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0}), read_vector<float>(result)));
}

TEST(op_eval, reduce_sum_keep_3d_eliminate_zero_dim)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 1, 2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the
    // right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((vector<float>{0, 0, 0, 0, 0, 0}), read_vector<float>(result)));
}

TEST(op_eval, reduce_sum_keep_3d_eliminate_zero_dim_int32)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{3, 1, 2};
    auto axes = make_shared<op::Constant>(element::i32, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<op::v1::ReduceSum>(A, axes, true), ParameterVector{A});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{});
    auto result = backend->create_tensor(element::i32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the
    // right value.
    copy_data(result, vector<int32_t>{2112, 2112, 2112, 2112, 2112, 2112});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<int32_t>{0, 0, 0, 0, 0, 0}), read_vector<int32_t>(result));
}

TEST(op_eval, reduce_sum_keep_dynamic)
{
    // Create a graph for f(x,axes:int32) = Sum(x,Convert<int64>(axes)).
    auto x = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto axes_i64 = make_shared<op::Convert>(axes, element::i64);

    auto sum = make_shared<op::v1::ReduceSum>(x, axes_i64, true);
    ASSERT_TRUE(sum->get_output_partial_shape(0).rank().is_dynamic());

    auto f = make_shared<Function>(NodeVector{sum}, ParameterVector{x, axes});

    auto backend = runtime::Backend::create("INTERPRETER", true);

    auto ex = backend->compile(f);

    auto t_r = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());

    std::vector<Shape> x_shapes{
        Shape{2, 3}, Shape{2, 3}, Shape{2, 3}, Shape{2, 3}, Shape{5}, Shape{5}};
    std::vector<std::vector<int32_t>> axeses{{}, {0}, {1}, {0, 1}, {}, {0}};
    std::vector<std::vector<float>> inputs{{1, 2, 3, 4, 5, 6},
                                           {1, 2, 3, 4, 5, 6},
                                           {1, 2, 3, 4, 5, 6},
                                           {1, 2, 3, 4, 5, 6},
                                           {1, 2, 3, 4, 5},
                                           {1, 2, 3, 4, 5}};
    std::vector<Shape> expected_result_shapes{
        Shape{2, 3}, Shape{1, 3}, Shape{2, 1}, Shape{1, 1}, Shape{5}, Shape{1}};
    std::vector<std::vector<float>> expected_results{
        {1, 2, 3, 4, 5, 6}, {5, 7, 9}, {6, 15}, {21}, {1, 2, 3, 4, 5}, {15}};

    for (size_t i = 0; i < x_shapes.size(); i++)
    {
        auto t_x = backend->create_tensor(element::f32, x_shapes[i]);
        auto t_axes = backend->create_tensor(element::i32, Shape{axeses[i].size()});

        copy_data(t_x, inputs[i]);
        copy_data(t_axes, axeses[i]);

        ex->call_with_validate({t_r}, {t_x, t_axes});

        ASSERT_EQ(t_r->get_shape(), expected_result_shapes[i]);

        auto results = read_vector<float>(t_r);

        ASSERT_TRUE(test::all_close_f(results, expected_results[i], MIN_FLOAT_TOLERANCE_BITS));
    }
}
