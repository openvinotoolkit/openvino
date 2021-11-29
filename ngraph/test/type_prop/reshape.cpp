// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, static_value_propagation)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);

    auto r = make_shared<op::v1::Reshape>(param, shape_of, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 2, 3}));
}

TEST(type_prop, interval_value_propagation)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension(1, 8), 2, 3});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);

    auto r = make_shared<op::v1::Reshape>(param, shape_of, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({Dimension(1, 8), 2, 3}));

    auto shape_of_opset1 = make_shared<op::v0::ShapeOf>(param);

    auto reshape = make_shared<op::v1::Reshape>(param, shape_of_opset1, false);

    ASSERT_EQ(reshape->get_element_type(), element::f32);
    ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({Dimension(1, 8), 2, 3}));
}

TEST(type_prop, static_value_propagation_through_gather)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto gather = make_shared<op::v1::Gather>(shape_of,
                                              op::Constant::create(element::i64, {3}, {2, 1, 0}),
                                              op::Constant::create(element::i64, {}, {0}));

    auto r = make_shared<op::v1::Reshape>(param, gather, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{3, 2, 1}));
}

TEST(type_prop, interval_value_propagation_through_gather)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension(1, 8), 2, 3});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto gather = make_shared<op::v1::Gather>(shape_of,
                                              op::Constant::create(element::i64, {3}, {2, 1, 0}),
                                              op::Constant::create(element::i64, {}, {0}));

    auto r = make_shared<op::v1::Reshape>(param, gather, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({3, 2, Dimension(1, 8)}));
}

TEST(type_prop, interval_value_propagation_through_consecutive_gathers)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension(1, 8), 2, 3});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto gather_1 = make_shared<op::v1::Gather>(shape_of,
                                                op::Constant::create(element::i64, {3}, {2, 1, 0}),
                                                op::Constant::create(element::i64, {}, {0}));

    auto gather_2 = make_shared<op::v1::Gather>(gather_1,
                                                op::Constant::create(element::i64, {3}, {1, 2, 0}),
                                                op::Constant::create(element::i64, {}, {0}));

    auto r = make_shared<op::v1::Reshape>(param, gather_2, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({2, Dimension(1, 8), 3}));
}

TEST(type_prop, interval_value_propagation_concatenated_gathers)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension(1, 8), 2, 3});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);

    auto gather_1 = make_shared<op::v1::Gather>(shape_of,
                                                op::Constant::create(element::i64, {}, {2}),
                                                op::Constant::create(element::i64, {}, {0}));
    auto dim_1 = make_shared<op::Unsqueeze>(gather_1, op::Constant::create(element::i64, {1}, {0}));

    auto gather_2 = make_shared<op::v1::Gather>(shape_of,
                                                op::Constant::create(element::i64, {}, {1}),
                                                op::Constant::create(element::i64, {}, {0}));
    auto tmp_dim_2 = make_shared<op::v1::Reshape>(
        gather_2, op::Constant::create(element::i64, {2}, {1, 1}), true);
    auto dim_2 =
        make_shared<op::v0::Squeeze>(tmp_dim_2, op::Constant::create(element::i64, {1}, {0}));

    auto gather_3 = make_shared<op::v1::Gather>(shape_of,
                                                op::Constant::create(element::i64, {}, {0}),
                                                op::Constant::create(element::i64, {}, {0}));
    auto dim_3 = make_shared<op::Unsqueeze>(gather_3, op::Constant::create(element::i64, {1}, {0}));

    auto shape = make_shared<op::Concat>(OutputVector{dim_1, dim_2, dim_3}, 0);
    auto r = make_shared<op::v1::Reshape>(param, shape, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({3, 2, Dimension(1, 8)}));
}

TEST(type_prop, interval_value_propagation_mul_div)
{
    auto param = make_shared<op::Parameter>(element::f32,
                                            PartialShape{Dimension(2, 8), Dimension(4, 16), 2});

    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<op::Convert>(shape_of, element::f32);
    auto mul = make_shared<op::v1::Multiply>(cast_fp,
                                             op::Constant::create(element::f32, {3}, {-2, 2, -4}));
    auto div =
        make_shared<op::v1::Divide>(mul, op::Constant::create(element::f32, {3}, {-2, 2, -4}));
    auto cast_int = make_shared<op::Convert>(div, element::i32);

    auto r = make_shared<op::v1::Reshape>(param, cast_int, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({Dimension(2, 8), Dimension(4, 16), 2}));
}

TEST(type_prop, interval_value_propagation_mul_div_lhs_scalar)
{
    auto param = make_shared<op::Parameter>(element::f32,
                                            PartialShape{Dimension(2, 8), Dimension(4, 16), 6});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<op::Convert>(shape_of, element::f32);
    auto mul = make_shared<op::v1::Multiply>(op::Constant::create(element::f32, {}, {2}), cast_fp);
    auto div = make_shared<op::v1::Divide>(mul, op::Constant::create(element::f32, {3}, {2, 1, 3}));
    auto cast_int = make_shared<op::Convert>(div, element::i32);

    auto r = make_shared<op::v1::Reshape>(param, cast_int, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({Dimension(2, 8), Dimension(8, 32), 4}));
}

TEST(type_prop, interval_value_propagation_mul_div_rhs_scalar)
{
    auto param = make_shared<op::Parameter>(element::f32,
                                            PartialShape{Dimension(2, 8), Dimension(4, 16), 6});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<op::Convert>(shape_of, element::f32);
    auto mul = make_shared<op::v1::Multiply>(cast_fp, op::Constant::create(element::f32, {}, {2}));
    auto div = make_shared<op::v1::Divide>(mul, op::Constant::create(element::f32, {3}, {2, 1, 3}));
    auto cast_int = make_shared<op::Convert>(div, element::i32);

    auto r = make_shared<op::v1::Reshape>(param, cast_int, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({Dimension(2, 8), Dimension(8, 32), 4}));
}

TEST(type_prop, interval_value_propagation_mul_div_lhs_1D)
{
    auto param = make_shared<op::Parameter>(element::f32,
                                            PartialShape{Dimension(2, 8), Dimension(4, 16), 6});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<op::Convert>(shape_of, element::f32);
    auto mul = make_shared<op::v1::Multiply>(op::Constant::create(element::f32, {1}, {2}), cast_fp);
    auto div = make_shared<op::v1::Divide>(mul, op::Constant::create(element::f32, {3}, {2, 1, 3}));
    auto cast_int = make_shared<op::Convert>(div, element::i32);

    auto r = make_shared<op::v1::Reshape>(param, cast_int, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({Dimension(2, 8), Dimension(8, 32), 4}));
}

TEST(type_prop, interval_value_propagation_mul_div_rhs_1D)
{
    auto param = make_shared<op::Parameter>(element::f32,
                                            PartialShape{Dimension(2, 8), Dimension(4, 16), 6});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<op::Convert>(shape_of, element::f32);
    auto mul = make_shared<op::v1::Multiply>(cast_fp, op::Constant::create(element::f32, {1}, {2}));
    auto div = make_shared<op::v1::Divide>(mul, op::Constant::create(element::f32, {3}, {2, 1, 3}));
    auto cast_int = make_shared<op::Convert>(div, element::i32);

    auto r = make_shared<op::v1::Reshape>(param, cast_int, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({Dimension(2, 8), Dimension(8, 32), 4}));
}

TEST(type_prop, interval_value_propagation_reduce)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension(1, 8), 2, 3});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto reduce_prod = make_shared<op::v1::ReduceProd>(
        shape_of, op::Constant::create(element::i64, {1}, {0}), true);
    auto r = make_shared<op::v1::Reshape>(param, reduce_prod, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape{Dimension(6, 48)});
}

TEST(type_prop, interval_value_propagation_reshape_zero_special_value)
{
    auto param = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension(1, 8), Dimension(16, 64), 3, Dimension(200, 400)});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);

    auto dim_021 = make_shared<op::v1::Gather>(shape_of,
                                               op::Constant::create(element::i64, {3}, {0, 2, 1}),
                                               op::Constant::create(element::i64, {}, {0}));
    auto dim_3 = op::Constant::create(element::i64, {1}, {0});

    auto shape = make_shared<op::Concat>(OutputVector{dim_021, dim_3}, 0);
    auto r = make_shared<op::v1::Reshape>(param, shape, true);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0),
              PartialShape({Dimension(1, 8), 3, Dimension(16, 64), Dimension(200, 400)}));
}

TEST(type_prop, interval_value_propagation_reshape_zero_minus_one_special_values)
{
    auto param = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension(1, 8), Dimension(16, 64), 6, Dimension(200, 400)});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);

    auto dim_0 = make_shared<op::v1::Gather>(shape_of,
                                             op::Constant::create(element::i64, {1}, {1}),
                                             op::Constant::create(element::i64, {}, {0}));
    auto dim_1 = op::Constant::create(element::i64, {1}, {0});
    auto dim_2 = op::Constant::create(element::i64, {1}, {-1});

    auto shape = make_shared<op::Concat>(OutputVector{dim_0, dim_1, dim_2}, 0);
    auto r = make_shared<op::v1::Reshape>(param, shape, true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0),
              PartialShape({Dimension(16, 64), Dimension(16, 64), Dimension(19, 1200)}));
}

TEST(type_prop, reshape_deduce_s2t)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {1}, Shape{1}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1}));
}

TEST(type_prop, reshape_deduce_s2m)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {2}, Shape{1, 1}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 1}));
}

TEST(type_prop, reshape_deduce_s2m3)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {3}, Shape{1, 1, 1}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 1, 1}));
}

TEST(type_prop, reshape_deduce_2d_to_1d)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {1}, Shape{12}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{12}));
}

TEST(type_prop, reshape_deduce_3d_to_1d)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {1}, Shape{60}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{60}));
}

TEST(type_prop, reshape_deduce_zero_special)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {3}, Shape{6, 2, 0}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{6, 2, 5}));
}

TEST(type_prop, reshape_deduce_wrong_output_shape)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    try
    {
        auto r = make_shared<op::v1::Reshape>(
            param, op::Constant::create(element::u64, {3}, Shape{3, 3, 3}), false);
        // Should have thrown, so fail if it didn't
        FAIL() << "No exception was thrown";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("is incompatible with input shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

//
// Input shape rank dynamic, so we should set the desired output shape
//
TEST(type_prop, reshape_partial_rank_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {4}, Shape{3, 1, 8, 2}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 8, 2}));
}

//
// Input shape rank static but input shape is dynamic, so should set desired output shape
//
TEST(type_prop, reshape_partial_rank_static)
{
    auto param_shape =
        PartialShape{Dimension::dynamic(), 6, Dimension::dynamic(), Dimension::dynamic()};
    auto param = make_shared<op::Parameter>(element::f32, param_shape);
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {4}, Shape{3, 1, 8, 2}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 8, 2}));
}

//
// Input shape rank static but input shape is dynamic, _but_ one of its static dimensions is zero,
// so should set desired output shape only if it also has zero elements.
//
TEST(type_prop, reshape_partial_rank_static_dynamic_but_zero_ok)
{
    auto param_shape =
        PartialShape{Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {4}, Shape{3, 1, 0, 2}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 0, 2}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_neg_zero)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1, 2});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{-1, 0}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{6, 1}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_neg)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1, 2});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{3, 2}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_neg_copy_input)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{3, 1}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_zero_one_neg)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {4}, std::vector<int64_t>{0, 0, 1, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{2, 2, 1, 3}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_neg_zero_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 1, 2});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{-1, 0}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 1}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_neg_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 1, 1});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 1}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_zero_one_neg_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {4}, std::vector<int64_t>{0, 0, 1, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{2, Dimension::dynamic(), 1, 3}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_neg_copy_input_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 1});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 1}));
}

TEST(type_prop, reshape_partial_rank_dynamic_special_zero)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {4}, std::vector<int64_t>{3, 1, 0, 2}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{3, 1, Dimension::dynamic(), 2}));
}

TEST(type_prop, reshape_partial_rank_dynamic_special_neg)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {4}, std::vector<int64_t>{3, -1, 0, 2}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0),
              (PartialShape{3, Dimension::dynamic(), Dimension::dynamic(), 2}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_zero_one_neg_dynamic_with_interval)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension(1, 3), 3});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {4}, std::vector<int64_t>{0, 0, 1, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{2, Dimension(1, 3), 1, 3}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_zero_one_neg_double_dynamic_with_interval)
{
    auto param = make_shared<op::Parameter>(element::f32,
                                            PartialShape{2, Dimension(1, 3), Dimension::dynamic()});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {4}, std::vector<int64_t>{0, 0, 1, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0),
              (PartialShape{2, Dimension(1, 3), 1, Dimension::dynamic()}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_neg_dynamic_with_interval)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension(1, 3)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{2, Dimension(1, 3)}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_neg_zero_dynamic_with_interval)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension(1, 3)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{-1, 0}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{2, Dimension(1, 3)}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_neg_zero_dynamic_with_interval_1)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension(1, 3), 2});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{-1, 0}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension(1, 3), 2}));
}

TEST(type_prop, reshape_pass_interval_dimension_through_minus_one)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension(1, 3), 2});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {3}, std::vector<int64_t>{0, -1, 2}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension(1, 3), 2}));
}

TEST(type_prop, reshape_multiply_interval_by_defined_dim_for_minus_one)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension(1, 3), 2});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension(2, 6)}));
}

TEST(type_prop, reshape_multiply_interval_by_interval_for_minus_one)
{
    auto param =
        make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension(1, 3), Dimension(1, 6)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension(1, 18)}));
}

TEST(type_prop, reshape_multiply_interval_by_interval_divide_by_defined_dim_for_minus_one)
{
    auto param = make_shared<op::Parameter>(element::f32,
                                            PartialShape{1, Dimension(1, 3), 3, Dimension(1, 6)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {3}, std::vector<int64_t>{0, -1, 3}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension(1, 18), 3}));
}

TEST(type_prop, reshape_multiply_interval_by_interval_divide_by_interval_for_minus_one)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{1, -1, Dimension(1, 6)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension::dynamic()}));
}

TEST(type_prop,
     reshape_multiply_interval_by_interval_divide_by_interval_for_minus_one_zero_included_in_input)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{1, -1, Dimension(0, 6)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension::dynamic()}));
}

TEST(type_prop, reshape_multiply_intervals_by_interval)
{
    auto param = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension(1, 2), Dimension(1, 3), Dimension(1, 4)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {1}, std::vector<int64_t>{-1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension(1, 24)}));
}

TEST(type_prop, reshape_multiply_intervals_by_interval_zero_included)
{
    auto param = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension(0, 2), Dimension(0, 3), Dimension(0, 4)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {1}, std::vector<int64_t>{-1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension(0, 24)}));
}

TEST(type_prop, reshape_to_zero_shape)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{0, 1});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {1}, std::vector<int64_t>{0}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_shape(0), (Shape{0}));
}

TEST(type_prop, reshape_to_zero_shape_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {1}, std::vector<int64_t>{0}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_shape(0), (Shape{0}));
}

TEST(type_prop, reshape_to_zero_shape_incorrect)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 1});
    ASSERT_THROW(
        make_shared<op::v1::Reshape>(
            param, op::Constant::create(element::i64, {1}, std::vector<int64_t>{0}), false),
        std::exception);
}

TEST(type_prop, reshape_to_zero)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 1});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {1}, std::vector<int64_t>{0}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_shape(0), (Shape{2}));
}

TEST(type_prop, reshape_to_scalar)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {}, std::vector<int64_t>{1}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_shape(0), (Shape{}));
}

TEST(type_prop, reshape_to_scalar_2)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {}, std::vector<int64_t>{1}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_shape(0), (Shape{}));
}

TEST(type_prop, reshape_to_scalar_3)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    ASSERT_THROW(
        make_shared<op::v1::Reshape>(
            param, op::Constant::create(element::i64, {}, std::vector<int64_t>{100}), false),
        std::exception);
}
