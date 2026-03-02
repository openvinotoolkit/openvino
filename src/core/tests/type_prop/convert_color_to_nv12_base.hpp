// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/op.hpp"

using namespace ov;
using namespace testing;

template <class T>
class ConvertToNV12BaseTest : public testing::Test {};

TYPED_TEST_SUITE_P(ConvertToNV12BaseTest);

TYPED_TEST_P(ConvertToNV12BaseTest, shape_inference_single_plane_default_ctor) {
    auto param_shape = PartialShape{5, 4, 6, 3};
    auto symbols = set_shape_symbols(param_shape);
    auto out_shape = PartialShape{5, 6, 6, 1};  // H*3/2 = 4*3/2 = 6
    auto param = std::make_shared<op::v0::Parameter>(element::f32, param_shape);

    auto op = std::make_shared<TypeParam>();
    op->set_argument(0, param);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->output(0).get_element_type(), element::f32);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, symbols[2], nullptr));
}

TYPED_TEST_P(ConvertToNV12BaseTest, shape_inference_single_plane) {
    auto param_shape = PartialShape{1, 4, 6, 3};
    auto symbols = set_shape_symbols(param_shape);
    auto out_shape = PartialShape{1, 6, 6, 1};
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    auto op = std::make_shared<TypeParam>(param);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->output(0).get_element_type(), element::u8);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, symbols[2], nullptr));
}

TYPED_TEST_P(ConvertToNV12BaseTest, shape_inference_single_plane_explicit_true) {
    auto param_shape = PartialShape{2, 8, 10, 3};
    auto out_shape = PartialShape{2, 12, 10, 1};
    auto param = std::make_shared<op::v0::Parameter>(element::f32, param_shape);
    auto op = std::make_shared<TypeParam>(param, true);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->output(0).get_element_type(), element::f32);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
}

TYPED_TEST_P(ConvertToNV12BaseTest, shape_inference_single_plane_dynamic) {
    auto param_shape = PartialShape::dynamic();
    auto out_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 1};
    auto param = std::make_shared<op::v0::Parameter>(element::f32, param_shape);
    auto op = std::make_shared<TypeParam>(param);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->output(0).get_element_type(), element::f32);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
}

TYPED_TEST_P(ConvertToNV12BaseTest, shape_inference_single_plane_dynamic_dims) {
    auto param_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), 8, 3};
    auto out_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), 8, 1};
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    auto op = std::make_shared<TypeParam>(param);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(op->output(0).get_element_type(), element::u8);
}

TYPED_TEST_P(ConvertToNV12BaseTest, shape_inference_single_plane_dynamic_type) {
    auto param_shape = PartialShape{1, 4, 6, 3};
    auto out_shape = PartialShape{1, 6, 6, 1};
    auto param = std::make_shared<op::v0::Parameter>(element::dynamic, param_shape);
    auto op = std::make_shared<TypeParam>(param);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(op->output(0).get_element_type(), element::dynamic);
}

TYPED_TEST_P(ConvertToNV12BaseTest, shape_inference_two_plane) {
    auto param_shape = PartialShape{1, 480, 640, 3};
    auto y_shape = PartialShape{1, 480, 640, 1};
    auto uv_shape = PartialShape{1, 240, 320, 2};
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    auto op = std::make_shared<TypeParam>(param, false);
    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_EQ(op->output(0).get_element_type(), element::u8);
    EXPECT_EQ(op->output(0).get_partial_shape(), y_shape);
    EXPECT_EQ(op->output(1).get_element_type(), element::u8);
    EXPECT_EQ(op->output(1).get_partial_shape(), uv_shape);
}

TYPED_TEST_P(ConvertToNV12BaseTest, shape_inference_two_plane_dynamic) {
    auto param_shape = PartialShape::dynamic();
    auto out_y_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 1};
    auto out_uv_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2};
    auto param = std::make_shared<op::v0::Parameter>(element::f32, param_shape);
    auto op = std::make_shared<TypeParam>(param, false);
    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_EQ(op->output(0).get_element_type(), element::f32);
    EXPECT_EQ(op->output(1).get_element_type(), element::f32);

    EXPECT_EQ(op->output(0).get_partial_shape(), out_y_shape);
    EXPECT_EQ(op->output(1).get_partial_shape(), out_uv_shape);
}

TYPED_TEST_P(ConvertToNV12BaseTest, shape_inference_two_plane_small) {
    auto param_shape = PartialShape{2, 4, 6, 3};
    auto y_shape = PartialShape{2, 4, 6, 1};
    auto uv_shape = PartialShape{2, 2, 3, 2};
    auto param = std::make_shared<op::v0::Parameter>(element::f32, param_shape);
    auto op = std::make_shared<TypeParam>(param, false);
    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_EQ(op->output(0).get_partial_shape(), y_shape);
    EXPECT_EQ(op->output(1).get_partial_shape(), uv_shape);
}

TYPED_TEST_P(ConvertToNV12BaseTest, error_channels_not_3) {
    auto param_shape = PartialShape{1, 4, 6, 1};  // C=1, not 3
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param), ov::AssertFailure);
}

TYPED_TEST_P(ConvertToNV12BaseTest, error_rank_5) {
    auto param_shape = PartialShape{1, 4, 6, 3, 1};  // 5 dims
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param), ov::AssertFailure);
}

TYPED_TEST_P(ConvertToNV12BaseTest, error_rank_3) {
    auto param_shape = PartialShape{4, 6, 3};  // 3 dims
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param), ov::AssertFailure);
}

TYPED_TEST_P(ConvertToNV12BaseTest, error_height_odd) {
    auto param_shape = PartialShape{1, 5, 6, 3};  // H=5 is odd
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param), ov::AssertFailure);
}

TYPED_TEST_P(ConvertToNV12BaseTest, error_width_odd) {
    auto param_shape = PartialShape{1, 4, 5, 3};  // W=5 is odd
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param), ov::AssertFailure);
}

TYPED_TEST_P(ConvertToNV12BaseTest, error_unsupported_type_i8) {
    auto param_shape = PartialShape{1, 4, 6, 3};
    auto param = std::make_shared<op::v0::Parameter>(element::i8, param_shape);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param), ov::AssertFailure);
}

TYPED_TEST_P(ConvertToNV12BaseTest, shape_inference_single_plane_interval_dims) {
    auto param_shape = PartialShape{{2, 10}, {4, 20}, 8, 3};
    auto symbols = set_shape_symbols(param_shape);
    auto out_shape = PartialShape{{2, 10}, {6, 30}, 8, 1};  // H*3/2
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    auto op = std::make_shared<TypeParam>(param);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(op->output(0).get_element_type(), element::u8);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, symbols[2], nullptr));
}

TYPED_TEST_P(ConvertToNV12BaseTest, shape_inference_two_plane_dynamic_dims) {
    auto param_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), 8, 3};
    auto expected_y_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), 8, 1};
    auto expected_uv_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), 4, 2};

    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    auto op = std::make_shared<TypeParam>(param, false);

    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_EQ(op->output(0).get_partial_shape(), expected_y_shape);
    EXPECT_EQ(op->output(1).get_partial_shape(), expected_uv_shape);
}

TYPED_TEST_P(ConvertToNV12BaseTest, shape_inference_two_plane_interval_dims_and_symbols) {
    auto param_shape = PartialShape{{2, 10}, {4, 20}, {6, 12}, 3};
    auto symbols = set_shape_symbols(param_shape);

    auto expected_y_shape = PartialShape{{2, 10}, {4, 20}, {6, 12}, 1};
    auto expected_uv_shape = PartialShape{{2, 10}, {2, 10}, {3, 6}, 2};

    auto param = std::make_shared<op::v0::Parameter>(element::f32, param_shape);
    auto op = std::make_shared<TypeParam>(param, false);

    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_EQ(op->output(0).get_partial_shape(), expected_y_shape);
    EXPECT_EQ(op->output(1).get_partial_shape(), expected_uv_shape);

    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(symbols[0], symbols[1], symbols[2], nullptr));

    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(1)), ElementsAre(symbols[0], nullptr, nullptr, nullptr));
}

TYPED_TEST_P(ConvertToNV12BaseTest, shape_inference_two_plane_dynamic_type) {
    auto param_shape = PartialShape{1, 480, 640, 3};
    auto out_y_shape = PartialShape{1, 480, 640, 1};
    auto out_uv_shape = PartialShape{1, 240, 320, 2};

    auto param = std::make_shared<op::v0::Parameter>(element::dynamic, param_shape);
    auto op = std::make_shared<TypeParam>(param, false);

    EXPECT_EQ(op->get_output_size(), 2);

    EXPECT_EQ(op->output(0).get_element_type(), element::dynamic);
    EXPECT_EQ(op->output(1).get_element_type(), element::dynamic);

    EXPECT_EQ(op->output(0).get_partial_shape(), out_y_shape);
    EXPECT_EQ(op->output(1).get_partial_shape(), out_uv_shape);
}

REGISTER_TYPED_TEST_SUITE_P(ConvertToNV12BaseTest,
                            shape_inference_single_plane_default_ctor,
                            shape_inference_single_plane,
                            shape_inference_single_plane_explicit_true,
                            shape_inference_single_plane_dynamic,
                            shape_inference_single_plane_dynamic_dims,
                            shape_inference_single_plane_dynamic_type,
                            shape_inference_two_plane,
                            shape_inference_two_plane_dynamic,
                            shape_inference_two_plane_small,
                            error_channels_not_3,
                            error_rank_5,
                            error_rank_3,
                            error_height_odd,
                            error_width_odd,
                            error_unsupported_type_i8,
                            shape_inference_single_plane_interval_dims,
                            shape_inference_two_plane_dynamic_dims,
                            shape_inference_two_plane_interval_dims_and_symbols,
                            shape_inference_two_plane_dynamic_type);
