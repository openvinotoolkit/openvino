// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reshape.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, static_value_propagation) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);

    auto r = make_shared<op::v1::Reshape>(param, shape_of, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 2, 3}));
}

TEST(type_prop, interval_value_propagation) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(1, 8), 2, 3});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);

    auto r = make_shared<op::v1::Reshape>(param, shape_of, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({Dimension(1, 8), 2, 3}));

    auto shape_of_opset1 = make_shared<op::v0::ShapeOf>(param);

    auto reshape = make_shared<op::v1::Reshape>(param, shape_of_opset1, false);

    ASSERT_EQ(reshape->get_element_type(), element::f32);
    ASSERT_EQ(reshape->get_output_partial_shape(0), PartialShape({Dimension(1, 8), 2, 3}));
}

TEST(type_prop, static_value_propagation_through_gather) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto gather = make_shared<op::v1::Gather>(shape_of,
                                              ov::op::v0::Constant::create(element::i64, {3}, {2, 1, 0}),
                                              ov::op::v0::Constant::create(element::i64, {}, {0}));

    auto r = make_shared<op::v1::Reshape>(param, gather, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{3, 2, 1}));
}

TEST(type_prop, interval_value_propagation_through_gather) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(1, 8), 2, 3});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto gather = make_shared<op::v1::Gather>(shape_of,
                                              ov::op::v0::Constant::create(element::i64, {3}, {2, 1, 0}),
                                              ov::op::v0::Constant::create(element::i64, {}, {0}));

    auto r = make_shared<op::v1::Reshape>(param, gather, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({3, 2, Dimension(1, 8)}));
}

TEST(type_prop, interval_value_propagation_through_consecutive_gathers) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(1, 8), 2, 3});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto gather_1 = make_shared<op::v1::Gather>(shape_of,
                                                ov::op::v0::Constant::create(element::i64, {3}, {2, 1, 0}),
                                                ov::op::v0::Constant::create(element::i64, {}, {0}));

    auto gather_2 = make_shared<op::v1::Gather>(gather_1,
                                                ov::op::v0::Constant::create(element::i64, {3}, {1, 2, 0}),
                                                ov::op::v0::Constant::create(element::i64, {}, {0}));

    auto r = make_shared<op::v1::Reshape>(param, gather_2, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({2, Dimension(1, 8), 3}));
}

TEST(type_prop, interval_value_propagation_concatenated_gathers) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(1, 8), 2, 3});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);

    auto gather_1 = make_shared<op::v1::Gather>(shape_of,
                                                ov::op::v0::Constant::create(element::i64, {}, {2}),
                                                ov::op::v0::Constant::create(element::i64, {}, {0}));
    auto dim_1 = make_shared<op::v0::Unsqueeze>(gather_1, ov::op::v0::Constant::create(element::i64, {1}, {0}));

    auto gather_2 = make_shared<op::v1::Gather>(shape_of,
                                                ov::op::v0::Constant::create(element::i64, {}, {1}),
                                                ov::op::v0::Constant::create(element::i64, {}, {0}));
    auto tmp_dim_2 =
        make_shared<op::v1::Reshape>(gather_2, ov::op::v0::Constant::create(element::i64, {2}, {1, 1}), true);
    auto dim_2 = make_shared<op::v0::Squeeze>(tmp_dim_2, ov::op::v0::Constant::create(element::i64, {1}, {0}));

    auto gather_3 = make_shared<op::v1::Gather>(shape_of,
                                                ov::op::v0::Constant::create(element::i64, {}, {0}),
                                                ov::op::v0::Constant::create(element::i64, {}, {0}));
    auto dim_3 = make_shared<op::v0::Unsqueeze>(gather_3, ov::op::v0::Constant::create(element::i64, {1}, {0}));

    auto shape = make_shared<op::v0::Concat>(OutputVector{dim_1, dim_2, dim_3}, 0);
    auto r = make_shared<op::v1::Reshape>(param, shape, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({3, 2, Dimension(1, 8)}));
}

TEST(type_prop, interval_value_propagation_mul_div) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(2, 8), Dimension(4, 16), 2});

    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<op::v0::Convert>(shape_of, element::f32);
    auto mul = make_shared<op::v1::Multiply>(cast_fp, ov::op::v0::Constant::create(element::f32, {3}, {-2, 2, -4}));
    auto div = make_shared<op::v1::Divide>(mul, ov::op::v0::Constant::create(element::f32, {3}, {-2, 2, -4}));
    auto cast_int = make_shared<op::v0::Convert>(div, element::i32);

    auto r = make_shared<op::v1::Reshape>(param, cast_int, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({Dimension(2, 8), Dimension(4, 16), 2}));
}

TEST(type_prop, interval_value_propagation_mul_div_rhs_shape) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32,
                                                    PartialShape{Dimension(1, 5), Dimension(0, 4), Dimension(2, 3)});

    auto shape_of = make_shared<op::v3::ShapeOf>(param);

    auto cast_fp = make_shared<op::v0::Convert>(shape_of, element::f32);
    auto mul = make_shared<op::v1::Multiply>(ov::op::v0::Constant::create(element::f32, {}, {2}), cast_fp);
    auto div = make_shared<op::v1::Divide>(ov::op::v0::Constant::create(element::f32, {3}, {10, 16, 12}), mul);
    auto cast_int = make_shared<op::v0::Convert>(div, element::i32);

    auto r = make_shared<op::v1::Reshape>(param, cast_int, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({Dimension(1, 5), Dimension(2, -1), Dimension(2, 3)}));
}

TEST(type_prop, interval_value_propagation_mul_div_lhs_scalar) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(2, 8), Dimension(4, 16), 6});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<op::v0::Convert>(shape_of, element::f32);
    auto mul = make_shared<op::v1::Multiply>(ov::op::v0::Constant::create(element::f32, {}, {2}), cast_fp);
    auto div = make_shared<op::v1::Divide>(mul, ov::op::v0::Constant::create(element::f32, {3}, {2, 1, 3}));
    auto cast_int = make_shared<op::v0::Convert>(div, element::i32);

    auto r = make_shared<op::v1::Reshape>(param, cast_int, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({Dimension(2, 8), Dimension(8, 32), 4}));
}

TEST(type_prop, interval_value_propagation_mul_div_rhs_scalar) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(2, 8), Dimension(4, 16), 6});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<op::v0::Convert>(shape_of, element::f32);
    auto mul = make_shared<op::v1::Multiply>(cast_fp, ov::op::v0::Constant::create(element::f32, {}, {2}));
    auto div = make_shared<op::v1::Divide>(mul, ov::op::v0::Constant::create(element::f32, {3}, {2, 1, 3}));
    auto cast_int = make_shared<op::v0::Convert>(div, element::i32);

    auto r = make_shared<op::v1::Reshape>(param, cast_int, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({Dimension(2, 8), Dimension(8, 32), 4}));
}

TEST(type_prop, interval_value_propagation_mul_lhs_1D_div) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(2, 8), Dimension(4, 16), 6});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<op::v0::Convert>(shape_of, element::f32);
    auto mul = make_shared<op::v1::Multiply>(ov::op::v0::Constant::create(element::f32, {1}, {2}), cast_fp);
    auto div = make_shared<op::v1::Divide>(mul, ov::op::v0::Constant::create(element::f32, {3}, {2, 1, 3}));
    auto cast_int = make_shared<op::v0::Convert>(div, element::i32);

    auto r = make_shared<op::v1::Reshape>(param, cast_int, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({Dimension(2, 8), Dimension(8, 32), 4}));
}

TEST(type_prop, interval_value_propagation_mul_rhs_1D_div) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(2, 8), Dimension(4, 16), 6});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<op::v0::Convert>(shape_of, element::f32);
    auto mul = make_shared<op::v1::Multiply>(cast_fp, ov::op::v0::Constant::create(element::f32, {1}, {2}));
    auto div = make_shared<op::v1::Divide>(mul, ov::op::v0::Constant::create(element::f32, {3}, {2, 1, 3}));
    auto cast_int = make_shared<op::v0::Convert>(div, element::i32);

    auto r = make_shared<op::v1::Reshape>(param, cast_int, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({Dimension(2, 8), Dimension(8, 32), 4}));
}

TEST(type_prop, interval_value_propagation_mul_div_lhs_1D) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(2, 8), Dimension(4, 16), 6});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto cast_fp = make_shared<op::v0::Convert>(shape_of, element::f32);
    auto mul = make_shared<op::v1::Multiply>(cast_fp, ov::op::v0::Constant::create(element::f32, {1}, {2}));
    auto div = make_shared<op::v1::Divide>(ov::op::v0::Constant::create(element::f32, {}, {192}), mul);
    auto cast_int = make_shared<op::v0::Convert>(div, element::i32);

    auto r = make_shared<op::v1::Reshape>(param, cast_int, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape({Dimension(12, 48), Dimension(6, 24), 16}));
}

TEST(type_prop, interval_value_propagation_reduce) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(1, 8), 2, 3});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);
    auto reduce_prod =
        make_shared<op::v1::ReduceProd>(shape_of, ov::op::v0::Constant::create(element::i64, {1}, {0}), true);
    auto r = make_shared<op::v1::Reshape>(param, reduce_prod, false);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), PartialShape{Dimension(6, 48)});
}

TEST(type_prop, interval_value_propagation_reshape_zero_special_value) {
    auto param =
        make_shared<ov::op::v0::Parameter>(element::f32,
                                           PartialShape{Dimension(1, 8), Dimension(16, 64), 3, Dimension(200, 400)});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);

    auto dim_021 = make_shared<op::v1::Gather>(shape_of,
                                               ov::op::v0::Constant::create(element::i64, {3}, {0, 2, 1}),
                                               ov::op::v0::Constant::create(element::i64, {}, {0}));
    auto dim_3 = ov::op::v0::Constant::create(element::i64, {1}, {0});

    auto shape = make_shared<op::v0::Concat>(OutputVector{dim_021, dim_3}, 0);
    auto r = make_shared<op::v1::Reshape>(param, shape, true);

    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0),
              PartialShape({Dimension(1, 8), 3, Dimension(16, 64), Dimension(200, 400)}));
}

TEST(type_prop, reshape_interval_value_propagation_reshape_zero_minus_one_special_values) {
    auto param =
        make_shared<op::v0::Parameter>(element::f32,
                                       PartialShape{Dimension(1, 8), Dimension(16, 64), 6, Dimension(200, 400)});
    auto shape_of = make_shared<op::v3::ShapeOf>(param);

    auto dim_0 = make_shared<op::v1::Gather>(shape_of,
                                             ov::op::v0::Constant::create(element::i64, {1}, {1}),
                                             ov::op::v0::Constant::create(element::i64, {}, {0}));
    auto dim_1 = ov::op::v0::Constant::create(element::i64, {1}, {0});
    auto dim_2 = ov::op::v0::Constant::create(element::i64, {1}, {-1});

    auto shape = make_shared<op::v0::Concat>(OutputVector{dim_0, dim_1, dim_2}, 0);
    auto r = make_shared<op::v1::Reshape>(param, shape, true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0),
              PartialShape({Dimension(16, 64), Dimension(16, 64), Dimension(19, 1200)}));
}

TEST(type_prop, reshape_deduce_s2t) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::v1::Reshape>(param, ov::op::v0::Constant::create(element::u64, {1}, Shape{1}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1}));
}

TEST(type_prop, reshape_deduce_s2m) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::v1::Reshape>(param, ov::op::v0::Constant::create(element::u64, {2}, Shape{1, 1}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 1}));
}

TEST(type_prop, reshape_deduce_s2m3) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto r =
        make_shared<op::v1::Reshape>(param, ov::op::v0::Constant::create(element::u64, {3}, Shape{1, 1, 1}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 1, 1}));
}

TEST(type_prop, reshape_deduce_2d_to_1d) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});
    auto r = make_shared<op::v1::Reshape>(param, ov::op::v0::Constant::create(element::u64, {1}, Shape{12}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{12}));
}

TEST(type_prop, reshape_deduce_3d_to_1d) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4, 5});
    auto r = make_shared<op::v1::Reshape>(param, ov::op::v0::Constant::create(element::u64, {1}, Shape{60}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{60}));
}

TEST(type_prop, reshape_deduce_zero_special) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4, 5});
    auto r = make_shared<op::v1::Reshape>(param, ov::op::v0::Constant::create(element::u64, {3}, Shape{6, 2, 0}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{6, 2, 5}));
}

//
// Input shape rank dynamic, so we should set the desired output shape
//
TEST(type_prop, reshape_partial_rank_dynamic) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto r =
        make_shared<op::v1::Reshape>(param, ov::op::v0::Constant::create(element::u64, {4}, Shape{3, 1, 8, 2}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 8, 2}));
}

//
// Input shape rank static but input shape is dynamic, so should set desired output shape
//
TEST(type_prop, reshape_partial_rank_static) {
    auto param_shape = PartialShape{Dimension::dynamic(), 6, Dimension::dynamic(), Dimension::dynamic()};
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, param_shape);
    auto r =
        make_shared<op::v1::Reshape>(param, ov::op::v0::Constant::create(element::u64, {4}, Shape{3, 1, 8, 2}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 8, 2}));
}

//
// Input shape rank static but input shape is dynamic, _but_ one of its static dimensions is zero,
// so should set desired output shape only if it also has zero elements.
//
TEST(type_prop, reshape_partial_rank_static_dynamic_but_zero_ok) {
    auto param_shape = PartialShape{Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto r =
        make_shared<op::v1::Reshape>(param, ov::op::v0::Constant::create(element::u64, {4}, Shape{3, 1, 0, 2}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 0, 2}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_neg_zero) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 1, 2});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {2}, std::vector<int64_t>{-1, 0}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{6, 1}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_neg) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 1, 2});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{3, 2}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_neg_copy_input) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 1});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{3, 1}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_zero_one_neg) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 3});
    auto r =
        make_shared<op::v1::Reshape>(param,
                                     ov::op::v0::Constant::create(element::i64, {4}, std::vector<int64_t>{0, 0, 1, -1}),
                                     true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{2, 2, 1, 3}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_neg_zero_dynamic) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 1, 2});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {2}, std::vector<int64_t>{-1, 0}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 1}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_neg_dynamic) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 1, 1});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 1}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_zero_one_neg_dynamic) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto r =
        make_shared<op::v1::Reshape>(param,
                                     ov::op::v0::Constant::create(element::i64, {4}, std::vector<int64_t>{0, 0, 1, -1}),
                                     true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{2, Dimension::dynamic(), 1, 3}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_neg_copy_input_dynamic) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 1});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 1}));
}

TEST(type_prop, reshape_partial_rank_dynamic_special_zero) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto r =
        make_shared<op::v1::Reshape>(param,
                                     ov::op::v0::Constant::create(element::i64, {4}, std::vector<int64_t>{3, 1, 0, 2}),
                                     true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{3, 1, Dimension::dynamic(), 2}));
}

TEST(type_prop, reshape_partial_rank_dynamic_special_neg) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto r =
        make_shared<op::v1::Reshape>(param,
                                     ov::op::v0::Constant::create(element::i64, {4}, std::vector<int64_t>{3, -1, 0, 2}),
                                     true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{3, Dimension::dynamic(), Dimension::dynamic(), 2}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_zero_one_neg_dynamic_with_interval) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, Dimension(1, 3), 3});
    auto r =
        make_shared<op::v1::Reshape>(param,
                                     ov::op::v0::Constant::create(element::i64, {4}, std::vector<int64_t>{0, 0, 1, -1}),
                                     true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{2, Dimension(1, 3), 1, 3}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_zero_one_neg_double_dynamic_with_interval) {
    auto param =
        make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, Dimension(1, 3), Dimension::dynamic()});
    auto r =
        make_shared<op::v1::Reshape>(param,
                                     ov::op::v0::Constant::create(element::i64, {4}, std::vector<int64_t>{0, 0, 1, -1}),
                                     true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{2, Dimension(1, 3), 1, Dimension::dynamic()}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_neg_dynamic_with_interval) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, Dimension(1, 3)});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{2, Dimension(1, 3)}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_neg_zero_dynamic_with_interval) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, Dimension(1, 3)});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {2}, std::vector<int64_t>{-1, 0}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{2, Dimension(1, 3)}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_neg_zero_dynamic_with_interval_1) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension(1, 3), 2});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {2}, std::vector<int64_t>{-1, 0}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension(1, 3), 2}));
}

TEST(type_prop, reshape_pass_interval_dimension_through_minus_one) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension(1, 3), 2});
    auto r =
        make_shared<op::v1::Reshape>(param,
                                     ov::op::v0::Constant::create(element::i64, {3}, std::vector<int64_t>{0, -1, 2}),
                                     true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension(1, 3), 2}));
}

TEST(type_prop, reshape_multiply_interval_by_defined_dim_for_minus_one) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension(1, 3), 2});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension(2, 6)}));
}

TEST(type_prop, reshape_multiply_interval_by_interval_for_minus_one) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension(1, 3), Dimension(1, 6)});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension(1, 18)}));
}

TEST(type_prop, reshape_multiply_interval_by_interval_divide_by_defined_dim_for_minus_one) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension(1, 3), 3, Dimension(1, 6)});
    auto r =
        make_shared<op::v1::Reshape>(param,
                                     ov::op::v0::Constant::create(element::i64, {3}, std::vector<int64_t>{0, -1, 3}),
                                     true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension(1, 18), 3}));
}

TEST(type_prop, reshape_multiply_interval_by_interval_divide_by_interval_for_minus_one) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, -1, Dimension(1, 6)});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension::dynamic()}));
}

TEST(type_prop, reshape_multiply_interval_by_interval_divide_by_interval_for_minus_one_zero_included_in_input) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, -1, Dimension(0, 6)});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension::dynamic()}));
}

TEST(type_prop, reshape_multiply_intervals_by_interval) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32,
                                                    PartialShape{Dimension(1, 2), Dimension(1, 3), Dimension(1, 4)});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {1}, std::vector<int64_t>{-1}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension(1, 24)}));
}

TEST(type_prop, reshape_multiply_intervals_by_interval_zero_included) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32,
                                                    PartialShape{Dimension(0, 2), Dimension(0, 3), Dimension(0, 4)});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {1}, std::vector<int64_t>{-1}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension(0, 24)}));
}

TEST(type_prop, reshape_to_zero_shape) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{0, 1});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {1}, std::vector<int64_t>{0}),
                                          false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_shape(0), (Shape{0}));
}

TEST(type_prop, reshape_to_zero_shape_dynamic) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {1}, std::vector<int64_t>{0}),
                                          false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_shape(0), (Shape{0}));
}

TEST(type_prop, reshape_to_zero_shape_incorrect) {
    auto param = make_shared<op::v0::Parameter>(element::f32, Shape{2, 1});

    OV_EXPECT_THROW(
        ignore = make_shared<op::v1::Reshape>(param,
                                              op::v0::Constant::create(element::i64, {1}, std::vector<int64_t>{0}),
                                              false),
        NodeValidationFailure,
        HasSubstr("Requested output shape [0] is incompatible with input shape"));
}

TEST(type_prop, reshape_to_zero) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 1});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {1}, std::vector<int64_t>{0}),
                                          true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_shape(0), (Shape{2}));
}

TEST(type_prop, reshape_to_scalar) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {}, std::vector<int64_t>{1}),
                                          false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_shape(0), (Shape{}));
}

TEST(type_prop, reshape_to_scalar_2) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::v1::Reshape>(param,
                                          ov::op::v0::Constant::create(element::i64, {}, std::vector<int64_t>{1}),
                                          false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_shape(0), (Shape{}));
}

TEST(type_prop, reshape_to_scalar_3) {
    auto param = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});

    OV_EXPECT_THROW(
        ignore = make_shared<op::v1::Reshape>(param,
                                              op::v0::Constant::create(element::i64, {}, std::vector<int64_t>{100}),
                                              false),
        NodeValidationFailure,
        HasSubstr("The value of scalar shape pattern should be equal to 1"));
}

TEST(type_prop, reshape_dynamic_shape_propagation_with_i32_precision) {
    auto param = make_shared<op::v0::Parameter>(element::f32, PartialShape{1, -1, -1});
    auto shape_of = make_shared<op::v3::ShapeOf>(param, element::i32);

    auto indices = op::v0::Constant::create(element::i32, {3}, {1, 2, 0});
    auto axis = op::v0::Constant::create(element::i32, {1}, {0});
    auto gather = make_shared<op::v1::Gather>(shape_of, indices, axis);

    auto reshape = make_shared<op::v1::Reshape>(param, gather, true);

    ASSERT_EQ(reshape->get_element_type(), element::f32);
    ASSERT_EQ(reshape->get_output_partial_shape(0), (PartialShape{-1, -1, 1}));
}

TEST(type_prop, reshape_dynamic_value_and_label_propagation) {
    Dimension marked_0 = Dimension(3);
    ov::DimensionTracker::set_label(marked_0, 10);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto param_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, target_0);
    auto shape_0 = std::make_shared<op::v0::ShapeOf>(param_0);

    const auto& et = element::i64;
    std::vector<int64_t> zero{0};
    const auto indices = make_shared<op::v0::Constant>(et, Shape{}, zero);
    const auto axis = make_shared<op::v0::Constant>(et, Shape{}, zero);
    const auto gather = make_shared<op::v7::Gather>(shape_0, indices, axis);

    const auto output_pattern = make_shared<op::v0::Constant>(et, Shape{1}, std::vector<int64_t>{-1});
    const auto unsqueeze = make_shared<op::v1::Reshape>(gather, output_pattern, false);

    auto bc = make_shared<op::v1::Broadcast>(param, unsqueeze);
    ASSERT_EQ(bc->get_shape(), (Shape{3}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, PartialShape({3}));
    EXPECT_THAT(get_shape_labels(output_shape), ElementsAre(10));
}

TEST(type_prop, reshape_label_shape_propagation_minus_one_variant_1) {
    PartialShape initial_shape = PartialShape{{1, 3}, 4, {1, 6}, 1, 3};
    ov::DimensionTracker::set_label(initial_shape[0], 10);
    ov::DimensionTracker::set_label(initial_shape[2], 12);

    auto input = make_shared<op::v0::Parameter>(element::f32, initial_shape);
    auto output_pattern = make_shared<op::v0::Constant>(element::i64, Shape{3}, std::vector<int64_t>{12, -1, 0});

    const auto reshape = make_shared<op::v1::Reshape>(input, output_pattern, true);

    auto output_shape = reshape->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, PartialShape({12, {1, 3}, {1, 6}}));
    EXPECT_THAT(get_shape_labels(output_shape), ElementsAre(ov::no_label, 10, 12));
}

TEST(type_prop, reshape_label_shape_propagation_minus_one_variant_2) {
    PartialShape initial_shape = PartialShape{4, -1, 2, 1, 3};
    ov::DimensionTracker::set_label(initial_shape[0], 10);
    ov::DimensionTracker::set_label(initial_shape[1], 11);

    auto input = make_shared<op::v0::Parameter>(element::f32, initial_shape);
    auto output_pattern = make_shared<op::v0::Constant>(element::i64, Shape{3}, std::vector<int64_t>{-1, 12, 2});

    const auto reshape = make_shared<op::v1::Reshape>(input, output_pattern, false);

    auto output_shape = reshape->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, PartialShape({-1, 12, 2}));
    EXPECT_THAT(get_shape_labels(output_shape), Each(ov::no_label));
}

TEST(type_prop, reshape_label_shape_propagation_minus_one_variant_3) {
    PartialShape initial_shape = PartialShape{{1, 3}, 4, {1, 6}, 1, 3};
    ov::DimensionTracker::set_label(initial_shape[0], 10);

    auto input = make_shared<op::v0::Parameter>(element::f32, initial_shape);
    auto output_pattern = make_shared<op::v0::Constant>(element::i64, Shape{3}, std::vector<int64_t>{1, -1, 12});

    const auto reshape = make_shared<op::v1::Reshape>(input, output_pattern, true);

    auto output_shape = reshape->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, PartialShape({1, {1, 18}, 12}));
    EXPECT_THAT(get_shape_labels(output_shape), Each(ov::no_label));
}

TEST(type_prop, reshape_label_shape_propagation_minus_one_variant_4) {
    PartialShape initial_shape = PartialShape{-1, 2, 2, 1, 3};
    ov::DimensionTracker::set_label(initial_shape[0], 10);

    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, initial_shape);
    auto output_pattern = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{12, -1});

    const auto reshape = make_shared<op::v1::Reshape>(input, output_pattern, true);

    auto output_shape = reshape->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, PartialShape({12, -1}));
    EXPECT_THAT(get_shape_labels(output_shape), ElementsAre(ov::no_label, 10));
}

TEST(type_prop, reshape_label_shape_propagation_minus_one_variant_5) {
    PartialShape initial_shape = PartialShape{-1, 2, 2, 1, 3};
    ov::DimensionTracker::set_label(initial_shape[0], 10);
    ov::DimensionTracker::set_label(initial_shape[1], 11);

    auto input = make_shared<op::v0::Parameter>(element::f32, initial_shape);
    auto output_pattern = make_shared<op::v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{-1, 12});

    const auto reshape = make_shared<op::v1::Reshape>(input, output_pattern, true);

    auto output_shape = reshape->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, PartialShape({-1, 12}));
    EXPECT_THAT(get_shape_labels(output_shape), Each(ov::no_label));
}

TEST(type_prop, reshape_label_shape_propagation_minus_one_variant_6) {
    PartialShape initial_shape = PartialShape{2, {2, 4}, 2, 3};
    ov::DimensionTracker::set_label(initial_shape[1], 10);

    auto input = make_shared<op::v0::Parameter>(element::f32, initial_shape);
    auto output_pattern =
        make_shared<op::v0::Constant>(element::i64, Shape{6}, std::vector<int64_t>{1, 4, 3, 1, 1, -1});

    const auto reshape = make_shared<op::v1::Reshape>(input, output_pattern, true);

    auto output_shape = reshape->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, PartialShape({1, 4, 3, 1, 1, {2, 4}}));
    EXPECT_THAT(get_shape_labels(output_shape),
                ElementsAre(ov::no_label, ov::no_label, ov::no_label, ov::no_label, ov::no_label, 10));
}

TEST(type_prop, reshape_label_shape_propagation_minus_one_variant_7) {
    PartialShape initial_shape = PartialShape{2, 3, 2, 1, 4};
    ov::DimensionTracker::set_label(initial_shape[1], 13);

    auto input = make_shared<op::v0::Parameter>(element::f32, initial_shape);
    auto output_pattern = make_shared<op::v0::Constant>(element::i64, Shape{5}, std::vector<int64_t>{4, 1, -1, 1, 4});

    const auto reshape = make_shared<op::v1::Reshape>(input, output_pattern, true);

    auto output_shape = reshape->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, PartialShape({4, 1, 3, 1, 4}));
    EXPECT_THAT(get_shape_labels(output_shape),
                ElementsAre(ov::no_label, ov::no_label, 13, ov::no_label, ov::no_label));
}

TEST(type_prop, reshape_when_pattern_has_static_shape_only) {
    auto param = make_shared<op::v0::Parameter>(element::f32, Shape{3, 4});
    auto shape_pattern = make_shared<op::v0::Parameter>(element::u64, PartialShape{3});
    auto r = make_shared<op::v1::Reshape>(param, shape_pattern, false);

    EXPECT_EQ(r->get_element_type(), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape::dynamic(3));
}

TEST(type_prop, reshape_when_pattern_has_interval_shape_only) {
    auto param = make_shared<op::v0::Parameter>(element::f32, Shape{3, 4});
    auto shape_pattern = make_shared<op::v0::Parameter>(element::u64, PartialShape{{1, 3}});
    auto r = make_shared<op::v1::Reshape>(param, shape_pattern, false);

    EXPECT_EQ(r->get_element_type(), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, reshape_when_pattern_has_scalar_shape_only) {
    auto param = make_shared<op::v0::Parameter>(element::f32, Shape{3, 4});
    auto shape_pattern = make_shared<op::v0::Parameter>(element::u64, PartialShape{});
    auto r = make_shared<op::v1::Reshape>(param, shape_pattern, false);

    EXPECT_EQ(r->get_element_type(), element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), PartialShape());
}

TEST(type_prop, reshape_label_propagation) {
    auto param_shape = PartialShape{{1, 2}, {2, 4}, 6, {2, 4}, 8};
    auto out_shape = PartialShape{{3, 5}, 0, 1, 0};
    set_shape_labels(param_shape, 10);
    set_shape_labels(out_shape, 20);

    const auto data = make_shared<op::v0::Parameter>(element::f32, param_shape);
    const auto out = make_shared<op::v0::Parameter>(element::f32, out_shape);
    const auto shape_of = make_shared<op::v3::ShapeOf>(out);
    auto dims_from_out_shape = make_shared<op::v1::Gather>(shape_of,
                                                           op::v0::Constant::create(element::i64, {4}, {0, 1, 2, 3}),
                                                           op::v0::Constant::create(element::i64, {}, {0}));
    const auto special_volume = op::v0::Constant::create(element::i64, {1}, {-1});
    const auto shape = make_shared<op::v0::Concat>(OutputVector{dims_from_out_shape, special_volume}, 0);

    const auto op = make_shared<op::v1::Reshape>(data, shape, true);

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{3, 5}, {2, 4}, 1, {2, 4}, {10, 32}}));
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), ElementsAre(20, 11, 22, 13, ov::no_label));
}

TEST(type_prop, reshape_label_propagation_minus_one_corner_case_zero_div_by_inf) {
    auto param_shape = PartialShape{0, 0};
    auto out_shape = PartialShape{-1, 2};
    set_shape_labels(param_shape, 10);
    set_shape_labels(out_shape, 20);

    const auto data = make_shared<op::v0::Parameter>(element::f32, param_shape);
    const auto out = make_shared<op::v0::Parameter>(element::f32, out_shape);
    const auto shape_of = make_shared<op::v3::ShapeOf>(out);
    auto dims_from_out_shape = make_shared<op::v1::Gather>(shape_of,
                                                           op::v0::Constant::create(element::i64, {2}, {0, 1}),
                                                           op::v0::Constant::create(element::i64, {}, {0}));
    const auto special_volume = op::v0::Constant::create(element::i64, {1}, {-1});
    const auto shape = make_shared<op::v0::Concat>(OutputVector{special_volume, dims_from_out_shape}, 0);

    const auto op = make_shared<op::v1::Reshape>(data, shape, true);

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, -1, 2}));
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), ElementsAre(ov::no_label, 20, 21));
}

TEST(type_prop, reshape_default_ctor) {
    auto param_shape = PartialShape{{1, 2}, {2, 4}, 6, {2, 4}, 8};
    auto out_shape = PartialShape{{3, 5}, 0, 1, 0};
    set_shape_labels(param_shape, 10);
    set_shape_labels(out_shape, 20);

    const auto data = make_shared<op::v0::Parameter>(element::f32, param_shape);
    const auto out = make_shared<op::v0::Parameter>(element::f32, out_shape);
    const auto shape_of = make_shared<op::v3::ShapeOf>(out);
    auto dims_from_out_shape = make_shared<op::v1::Gather>(shape_of,
                                                           op::v0::Constant::create(element::i64, {4}, {0, 1, 2, 3}),
                                                           op::v0::Constant::create(element::i64, {}, {0}));
    const auto special_volume = op::v0::Constant::create(element::i64, {1}, {-1});
    const auto shape = make_shared<op::v0::Concat>(OutputVector{dims_from_out_shape, special_volume}, 0);

    const auto op = make_shared<op::v1::Reshape>();
    op->set_arguments(OutputVector{data, shape});
    op->set_special_zero(true);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{3, 5}, {2, 4}, 1, {2, 4}, {10, 32}}));
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), ElementsAre(20, 11, 22, 13, ov::no_label));
}

TEST(type_prop, reshape_data_2d_dynamic_pattern_has_minus_one_i32) {
    const auto param = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(2));
    const auto op = make_shared<op::v1::Reshape>(param, op::v0::Constant::create(element::i32, {2}, {-1, 4}), false);

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, 4}));
}

TEST(type_prop, reshape_deduce_wrong_output_shape) {
    auto param = make_shared<op::v0::Parameter>(element::f32, Shape{3, 4, 5});

    OV_EXPECT_THROW(
        ignore = make_shared<op::v1::Reshape>(param, op::v0::Constant::create(element::u64, {3}, {3, 3, 3}), false),
        NodeValidationFailure,
        HasSubstr("is incompatible with input shape"));
}

TEST(type_prop, reshape_pattern_shape_not_1d) {
    auto param = make_shared<op::v0::Parameter>(element::f32, Shape{3, 4, 5});

    OV_EXPECT_THROW(
        ignore =
            make_shared<op::v1::Reshape>(param, op::v0::Constant::create(element::u64, {3, 1}, Shape{3, 5, 4}), false),
        NodeValidationFailure,
        HasSubstr("Pattern shape must have rank 1 or be empty"));
}

TEST(type_prop, reshape_multiple_minus_one_no_special_zero) {
    const auto data = make_shared<op::v0::Parameter>(element::f32, PartialShape{{1, 2}, {2, 4}, 6, {2, 4}, 8});

    OV_EXPECT_THROW(
        ignore = make_shared<op::v1::Reshape>(data, op::v0::Constant::create(element::i64, {3}, {-1, 5, -1}), false),
        NodeValidationFailure,
        HasSubstr("More than one dimension has size of -1"));
}

TEST(type_prop, reshape_multiple_minus_one_special_zero_set) {
    const auto data = make_shared<op::v0::Parameter>(element::f32, PartialShape{{1, 2}, {2, 4}, 6, {2, 4}, 8});

    OV_EXPECT_THROW(
        ignore = make_shared<op::v1::Reshape>(data, op::v0::Constant::create(element::i64, {3}, {-1, 5, -1}), true),
        NodeValidationFailure,
        HasSubstr("More than one dimension has size of -1"));
}

TEST(type_prop, reshape_special_zero_out_of_data_rank) {
    const auto data = make_shared<op::v0::Parameter>(element::f32, PartialShape{{1, 2}, {2, 4}, 8});

    OV_EXPECT_THROW(
        ignore = make_shared<op::v1::Reshape>(data, op::v0::Constant::create(element::i64, {4}, {5, 1, 1, 0}), true),
        NodeValidationFailure,
        HasSubstr("'0' dimension is out of range"));
}

TEST(type_prop, reshape_special_zero_cannot_div) {
    const auto data = make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 5, 4});

    OV_EXPECT_THROW(
        ignore = make_shared<op::v1::Reshape>(data, op::v0::Constant::create(element::i64, {3}, {10, -1, 3}), false),
        NodeValidationFailure,
        HasSubstr("Non-'-1' output dimensions do not evenly divide the input dimensions"));
}

TEST(type_prop, reshape_zero_dim_in_output_pattern_but_not_in_data_shape) {
    const auto data = make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 5, 4});

    OV_EXPECT_THROW(
        ignore = make_shared<op::v1::Reshape>(data, op::v0::Constant::create(element::i64, {3}, {5, 0, -1}), false),
        NodeValidationFailure,
        HasSubstr("Cannot infer '-1' dimension with zero-size output dimension unless at least one input dimension is "
                  "also zero-size"));
}
