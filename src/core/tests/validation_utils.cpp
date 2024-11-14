// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/util/common_util.hpp"

TEST(get_constant_from_source, invalidation_check) {
    auto a = ov::opset8::Constant::create(ov::element::i64, {100}, {123});
    auto b = ov::opset8::Constant::create(ov::element::i64, {1}, {123});
    auto div = std::make_shared<ov::opset8::Divide>(a, b);
    auto s = std::make_shared<ov::opset8::ShapeOf>(a);
    auto r = std::make_shared<ov::opset8::Reshape>(div, s, true);
    auto tmp_consumer = std::make_shared<ov::opset8::ShapeOf>(s);

    ASSERT_TRUE(ov::util::get_constant_from_source(r));

    ASSERT_TRUE(r->get_output_tensor(0).get_lower_value());
    ASSERT_TRUE(r->get_output_tensor(0).get_upper_value());

    ASSERT_TRUE(s->get_output_tensor(0).get_lower_value());
    ASSERT_TRUE(s->get_output_tensor(0).get_upper_value());

    ASSERT_TRUE(b->get_output_tensor(0).get_lower_value());
    ASSERT_TRUE(b->get_output_tensor(0).get_upper_value());

    ASSERT_TRUE(a->get_output_tensor(0).get_lower_value());
    ASSERT_TRUE(a->get_output_tensor(0).get_upper_value());

    ASSERT_FALSE(div->get_output_tensor(0).get_lower_value());
    ASSERT_FALSE(div->get_output_tensor(0).get_upper_value());
}

TEST(get_constant_from_source, extract_static_dim_from_dynamic_shape_check) {
    auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, 128});
    auto shape = std::make_shared<ov::opset8::ShapeOf>(data);
    auto one = ov::opset8::Constant::create(ov::element::i64, {1}, {1});
    auto zero = ov::opset8::Constant::create(ov::element::i64, {1}, {0});
    const auto extract_static_dimension = std::make_shared<ov::opset8::Gather>(shape, one, zero);

    ASSERT_TRUE(ov::util::get_constant_from_source(extract_static_dimension));

    ASSERT_TRUE(extract_static_dimension->get_output_tensor(0).get_lower_value());
    ASSERT_TRUE(extract_static_dimension->get_output_tensor(0).get_upper_value());
}

TEST(get_constant_from_source, return_nullptr_for_empty_output) {
    auto res = ov::util::get_constant_from_source(ov::Output<ov::Node>());
    ASSERT_EQ(res, nullptr);
}

TEST(constantfold_subgraph, split) {
    std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8};
    auto constant = ov::opset8::Constant::create(ov::element::f32, ov::Shape{input.size()}, input);
    auto mul = std::make_shared<ov::opset8::Multiply>(constant,
                                                      ov::opset8::Constant::create(ov::element::f32, ov::Shape{}, {1}));
    auto shape = std::make_shared<ov::opset8::ShapeOf>(mul);
    auto len_0 =
        std::make_shared<ov::opset8::Divide>(shape, ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {2}));
    auto len_1 = std::make_shared<ov::opset8::Subtract>(shape, len_0);
    auto lenghts = std::make_shared<ov::opset8::Concat>(ov::OutputVector{len_0, len_1}, 0);
    auto axis = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto split = std::make_shared<ov::opset8::VariadicSplit>(mul, axis, lenghts);
    std::vector<float> expected(std::next(input.begin(), input.size() / 2), input.end());
    auto ret = ov::util::constantfold_subgraph(split->output(1));
    ASSERT_NE(ret, nullptr);
    auto actual = ret->cast_vector<float>();
    ASSERT_EQ(expected, actual);
}

TEST(constantfold_subgraph, shapeof) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 3, -1});
    auto shapeof = std::make_shared<ov::op::v3::ShapeOf>(param);
    auto zero = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
    auto one = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
    auto two = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {2});
    auto stop = std::make_shared<ov::op::v8::Slice>(shapeof, one /*start*/, two /*stop*/, one /*step*/, zero /*axis*/);
    auto slice = std::make_shared<ov::op::v8::Slice>(param, one /*start*/, stop, one /*step*/, one /*axis*/);

    auto ret = ov::util::constantfold_subgraph(stop);
    ASSERT_NE(ret, nullptr);
    auto actual = ret->cast_vector<int64_t>();
    std::vector<int64_t> expected{3};
    ASSERT_EQ(expected, actual);
}

namespace ov {
namespace test {

using op::v0::Concat;
using op::v0::Constant;
using op::v0::Parameter;

using testing::HasSubstr;
using testing::Values;

using NormalizeAxisTest = testing::Test;

TEST_F(NormalizeAxisTest, try_normalize_dynamic_rank) {
    OV_EXPECT_THROW(util::try_normalize_axis(3, Rank(4, 5)), Exception, testing::_);
}

TEST_F(NormalizeAxisTest, try_normalize_invalid_axis) {
    OV_EXPECT_THROW(util::try_normalize_axis(-7, Rank(6)),
                    Exception,
                    HasSubstr("Axis -7 out of the tensor rank range [-6, 5]"));
    OV_EXPECT_THROW(util::try_normalize_axis(6, Rank(6)),
                    Exception,
                    HasSubstr("Axis 6 out of the tensor rank range [-6, 5]"));
}

TEST_F(NormalizeAxisTest, try_normalize_invalid_axis_with_node_description) {
    auto node = Parameter(element::i32, PartialShape::dynamic(2));
    node.set_friendly_name("My node");
    const auto n_rank = node.get_output_partial_shape(0).rank();

    OV_EXPECT_THROW(util::try_normalize_axis(-7, n_rank, node),
                    Exception,
                    HasSubstr("My node':\nAxis -7 out of the tensor rank range [-2, 1]"));
    OV_EXPECT_THROW(util::try_normalize_axis(2, n_rank, node),
                    Exception,
                    HasSubstr("My node':\nAxis 2 out of the tensor rank range [-2, 1]"));
}

TEST_F(NormalizeAxisTest, validate_node_axis) {
    const auto shape = PartialShape{2, 4, 5, 6};
    const auto p = std::make_shared<Parameter>(element::i32, shape);
    const auto node = std::make_shared<Concat>(NodeVector{1, p}, -1);

    OV_EXPECT_THROW(util::validate_axis(-5, shape.rank(), *node.get()),
                    NodeValidationFailure,
                    HasSubstr("Axis -5 out of the tensor rank range [-4, 3]"));
    OV_EXPECT_THROW(util::validate_axis(4, shape.rank(), *node),
                    NodeValidationFailure,
                    HasSubstr("Axis 4 out of the tensor rank range [-4, 3]"));
}

TEST_F(NormalizeAxisTest, validate_axes_correct) {
    const auto axes = std::vector<int64_t>{-2, 4, 3, 0, -1};
    const auto node = Parameter(element::i32, PartialShape::dynamic(6));

    OV_ASSERT_NO_THROW(util::validate_axes(axes, Rank(6), node));
}

TEST_F(NormalizeAxisTest, validate_axes_in_correct) {
    const auto axes = std::vector<int64_t>{-2, 4, 3, 0, -1};
    const auto node = Parameter(element::i16, PartialShape::dynamic(4));

    OV_EXPECT_THROW(util::validate_axes(axes, Rank(3), node),
                    Exception,
                    HasSubstr("Axis 4 out of the tensor rank range [-3, 2]"));
}

TEST_F(NormalizeAxisTest, normalize_axes) {
    auto axes = std::vector<int64_t>{-2, 4, 3, 0, -1};
    const auto exp_axes = std::vector<int64_t>{4, 4, 3, 0, 5};

    util::normalize_axes(axes, 6);

    EXPECT_EQ(exp_axes, axes);
}

TEST_F(NormalizeAxisTest, try_normalize_axes) {
    auto axes = std::vector<int64_t>{-2, 4, 3, 0, -1};
    const auto exp_axes = std::vector<int64_t>{4, 4, 3, 0, 5};
    const auto node = Parameter(element::i64, PartialShape::dynamic(6));

    util::try_normalize_axes(axes, Rank(6), node);

    EXPECT_EQ(exp_axes, axes);
}

TEST_F(NormalizeAxisTest, try_get_normalize_axis_vector) {
    const auto const_axes = Constant::create(element::i64, Shape{6}, {-2, 1, 0, -3, 2, -1});

    const auto axes = util::try_get_normalized_axis_vector(const_axes->get_tensor_view(), Rank(4), *const_axes);

    EXPECT_EQ(AxisVector({2, 1, 0, 1, 2, 3}), axes);
}

TEST_F(NormalizeAxisTest, try_get_normalize_axis_vector_fail) {
    const auto const_axes = Constant::create(element::i64, Shape{6}, {-2, 1, 0, -3, 2, -1});

    OV_EXPECT_THROW(util::try_get_normalized_axis_vector(const_axes->get_tensor_view(), Rank(2), *const_axes),
                    Exception,
                    HasSubstr("Axis -3 out of the tensor rank range [-2, 1]"));
}

TEST_F(NormalizeAxisTest, try_get_normalize_axis_set) {
    const auto const_axes = Constant::create(element::i64, Shape{6}, {-2, 1, 0, -3, 2, -1});

    const auto axes = util::try_get_normalized_axis_set(const_axes->get_tensor_view(), Rank(4), *const_axes);

    EXPECT_EQ(AxisSet({0, 1, 2, 3}), axes);
}

TEST_F(NormalizeAxisTest, try_get_normalize_axis_set_fail) {
    const auto const_axes = Constant::create(element::i32, Shape{6}, {-2, 1, 0, -3, 2, -1});

    OV_EXPECT_THROW(util::try_get_normalized_axis_set(const_axes->get_tensor_view(), Rank(2), *const_axes),
                    Exception,
                    HasSubstr("Axis -3 out of the tensor rank range [-2, 1]"));
}

using NormalizeAxisParam = std::tuple<int64_t,  // axis
                                      size_t,   // normalized axis
                                      Rank      // Rank
                                      >;

class NormalizeAxisTestP : public NormalizeAxisTest, public testing::WithParamInterface<NormalizeAxisParam> {};

INSTANTIATE_TEST_SUITE_P(positive_axis,
                         NormalizeAxisTestP,
                         Values(NormalizeAxisParam(0, 0, Rank{0}),
                                NormalizeAxisParam{0, 0, Rank{5}},
                                NormalizeAxisParam{3, 3, Rank{5}},
                                NormalizeAxisParam{4, 4, Rank{5}}),
                         testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(negative_axis,
                         NormalizeAxisTestP,
                         Values(NormalizeAxisParam{-5, 0, Rank{5}},
                                NormalizeAxisParam{-3, 2, Rank{5}},
                                NormalizeAxisParam{-1, 4, Rank{5}}),
                         testing::PrintToStringParamName());

TEST_P(NormalizeAxisTestP, is_axis_valid) {
    const auto& axis = std::get<0>(GetParam());
    const auto& rank = std::get<2>(GetParam());

    EXPECT_TRUE(util::is_axis_valid(axis, rank.get_length()));
}

TEST_P(NormalizeAxisTestP, is_axis_valid_scalar_rank) {
    const auto& axis = std::get<0>(GetParam());

    if (axis) {
        EXPECT_FALSE(util::is_axis_valid(axis, 0));
    }
}

TEST_P(NormalizeAxisTestP, validate_axis) {
    const auto axes_node = Parameter(element::i64, PartialShape::dynamic(5));
    const auto& axis = std::get<0>(GetParam());

    EXPECT_NO_THROW(util::validate_axis(axis, axes_node.get_output_partial_shape(0).rank(), axes_node));
}

TEST_P(NormalizeAxisTestP, normalize_axis_rank) {
    const auto& axis = std::get<0>(GetParam());
    const auto& exp_axis = std::get<1>(GetParam());
    const auto& rank = std::get<2>(GetParam());

    EXPECT_EQ(exp_axis, util::normalize_axis(axis, rank.get_length()));
}

TEST_P(NormalizeAxisTestP, try_normalize_axis) {
    const auto& axis = std::get<0>(GetParam());
    const auto& exp_axis = std::get<1>(GetParam());
    const auto& rank = std::get<2>(GetParam());

    EXPECT_EQ(exp_axis, util::try_normalize_axis(axis, rank));
}
}  // namespace test
}  // namespace ov
