// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/op.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace ov;
using namespace testing;

template <class T>
class ConvertNV12BaseTest : public testing::Test {};

TYPED_TEST_SUITE_P(ConvertNV12BaseTest);

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_default_ctor_single_plane) {
    auto param_shape = PartialShape{5, 3, 2, 1};
    auto symbols = set_shape_symbols(param_shape);
    auto out_shape = PartialShape{5, 2, 2, 3};
    auto param = std::make_shared<op::v0::Parameter>(element::f32, param_shape);

    auto op = std::make_shared<TypeParam>();
    op->set_argument(0, param);
    op->validate_and_infer_types();

    EXPECT_EQ(op->output(0).get_element_type(), element::f32);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, symbols[2], nullptr));
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_single_tensor) {
    auto param_shape = PartialShape{5, 3, 2, 1};
    auto symbols = set_shape_symbols(param_shape);
    auto out_shape = PartialShape{5, 2, 2, 3};
    auto param = std::make_shared<op::v0::Parameter>(element::f32, param_shape);
    auto op = std::make_shared<TypeParam>(param);
    EXPECT_EQ(op->output(0).get_element_type(), element::f32);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, symbols[2], nullptr));
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_single_tensor_dynamic) {
    auto param_shape = PartialShape::dynamic();
    auto out_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 3};
    auto param = std::make_shared<op::v0::Parameter>(element::f32, param_shape);
    auto op = std::make_shared<TypeParam>(param);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(op->output(0).get_element_type(), element::f32);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_single_tensor_dynamic_dims) {
    auto param_shape = PartialShape{Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()};
    auto out_shape = PartialShape{Dimension::dynamic(), 2, Dimension::dynamic(), 3};
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    auto op = std::make_shared<TypeParam>(param);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(op->output(0).get_element_type(), element::u8);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_single_tensor_dynamic_height) {
    auto param_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), 8, Dimension::dynamic()};
    auto out_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), 8, 3};
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    auto op = std::make_shared<TypeParam>(param);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(op->output(0).get_element_type(), element::u8);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_single_tensor_dynamic_type) {
    auto param_shape = PartialShape{1, 6, 8, 1};
    auto out_shape = PartialShape{1, 4, 8, 3};
    auto param = std::make_shared<op::v0::Parameter>(element::dynamic, param_shape);
    auto op = std::make_shared<TypeParam>(param);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(op->output(0).get_element_type(), element::dynamic);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_single_tensor_error_channels) {
    auto param_shape = PartialShape{1, 3, 4, 2};  // shall be 1 channel, not 2
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_single_tensor_error_dims_5) {
    auto param_shape = PartialShape{1, 3, 3, 1, 1};  // must be 4 dimensions
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_single_tensor_error_dims_3) {
    auto param_shape = PartialShape{640, 480, 1};  // must be 4 dimensions
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_single_tensor_error_height) {
    auto param_shape = PartialShape{1, 4, 6, 1};  // height = 4, can't split to Y and UV
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_single_tensor_error_width_odd) {
    auto param_shape = PartialShape{1, 6, 5, 1};  // width is odd, can't split to U and V
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_single_tensor_error_i8) {
    auto param_shape = PartialShape{1, 640, 480, 1};
    auto param = std::make_shared<op::v0::Parameter>(element::i8, param_shape);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_single_interval_dims_and_symbols) {
    auto param_shape = PartialShape{{2, 20}, {5, 10}, 8, 1};
    auto symbols = set_shape_symbols(param_shape);
    auto out_shape = PartialShape{{2, 20}, {4, 6}, 8, 3};
    auto param = std::make_shared<op::v0::Parameter>(element::u8, param_shape);
    auto op = std::make_shared<TypeParam>(param);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(op->output(0).get_element_type(), element::u8);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, symbols[2], nullptr));
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_default_ctor_2_planes) {
    auto param_shape_y = PartialShape{10, 480, 640, 1};
    auto param_shape_uv = PartialShape{10, 240, 320, 2};
    auto out_shape = PartialShape{10, 480, 640, 3};
    auto param_y = std::make_shared<op::v0::Parameter>(element::u8, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(element::u8, param_shape_uv);

    auto op = std::make_shared<TypeParam>();
    op->set_arguments(OutputVector{param_y, param_uv});
    op->validate_and_infer_types();
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(op->output(0).get_element_type(), element::u8);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_simple) {
    auto param_shape_y = PartialShape{10, 480, 640, 1};
    auto param_shape_uv = PartialShape{10, 240, 320, 2};
    auto out_shape = PartialShape{10, 480, 640, 3};
    auto param_y = std::make_shared<op::v0::Parameter>(element::u8, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(element::u8, param_shape_uv);
    auto op = std::make_shared<TypeParam>(param_y, param_uv);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(op->output(0).get_element_type(), element::u8);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_dynamic) {
    auto param_shape_y = PartialShape::dynamic();
    auto param_shape_uv = PartialShape::dynamic();
    auto out_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 3};
    auto param_y = std::make_shared<op::v0::Parameter>(element::f32, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(element::f32, param_shape_uv);
    auto op = std::make_shared<TypeParam>(param_y, param_uv);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(op->output(0).get_element_type(), element::f32);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_y_dynamic) {
    auto param_shape_y = PartialShape::dynamic();
    auto param_shape_uv = PartialShape{1, 3, 2, 2};
    auto out_shape = PartialShape{1, 6, 4, 3};
    auto param_y = std::make_shared<op::v0::Parameter>(element::bf16, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(element::bf16, param_shape_uv);
    auto op = std::make_shared<TypeParam>(param_y, param_uv);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(op->output(0).get_element_type(), element::bf16);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_uv_dynamic) {
    auto param_shape_y = PartialShape{1, 4, 4, 1};
    auto param_shape_uv = PartialShape::dynamic();
    auto out_shape = PartialShape{1, 4, 4, 3};
    auto param_y = std::make_shared<op::v0::Parameter>(element::f16, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(element::f16, param_shape_uv);
    auto op = std::make_shared<TypeParam>(param_y, param_uv);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(op->output(0).get_element_type(), element::f16);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_dynamic_types) {
    auto param_shape_y = PartialShape{1, 4, 4, 1};
    auto param_shape_uv = PartialShape{1, 2, 2, 2};
    auto y_symbols = set_shape_symbols(param_shape_y);
    auto uv_symbols = set_shape_symbols(param_shape_uv);

    auto out_shape = PartialShape{1, 4, 4, 3};
    auto y_type = element::dynamic;
    auto uv_type = element::dynamic;
    auto out_type = element::dynamic;
    auto param_y = std::make_shared<op::v0::Parameter>(y_type, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(uv_type, param_shape_uv);
    auto op = std::make_shared<TypeParam>(param_y, param_uv);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(op->output(0).get_element_type(), out_type);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(y_symbols[0], y_symbols[1], y_symbols[2], nullptr));
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_uv_type) {
    auto param_shape_y = PartialShape{1, 4, 4, 1};
    auto param_shape_uv = PartialShape{1, 2, 2, 2};
    auto symbols = set_shape_symbols(param_shape_uv);

    auto out_shape = PartialShape{1, 4, 4, 3};
    auto y_type = element::dynamic;
    auto uv_type = element::f64;
    auto out_type = element::f64;
    auto param_y = std::make_shared<op::v0::Parameter>(y_type, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(uv_type, param_shape_uv);
    auto op = std::make_shared<TypeParam>(param_y, param_uv);
    EXPECT_EQ(op->output(0).get_partial_shape(), out_shape);
    EXPECT_EQ(op->output(0).get_element_type(), out_type);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr, nullptr, nullptr));
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_error_type_mismatch) {
    auto param_y = std::make_shared<op::v0::Parameter>(element::u8, PartialShape::dynamic());
    auto param_uv = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param_y, param_uv), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_error_uv_type) {
    auto param_y = std::make_shared<op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param_uv = std::make_shared<op::v0::Parameter>(element::i8, PartialShape::dynamic());
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param_y, param_uv), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_error_5dims) {
    auto param_shape_y = PartialShape::dynamic();
    auto param_shape_uv = PartialShape{2, 2, 2, 2, 2};
    auto param_y = std::make_shared<op::v0::Parameter>(element::u8, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(element::u8, param_shape_uv);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param_y, param_uv), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_error_3dims) {
    auto param_shape_y = PartialShape::dynamic();
    auto param_shape_uv = PartialShape{2, 2, 2};
    auto param_y = std::make_shared<op::v0::Parameter>(element::u8, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(element::u8, param_shape_uv);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param_y, param_uv), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_error_batch) {
    auto param_shape_y = PartialShape{2, 480, 640, 1};
    auto param_shape_uv = PartialShape{1, 240, 320, 2};
    auto param_y = std::make_shared<op::v0::Parameter>(element::u8, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(element::u8, param_shape_uv);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param_y, param_uv), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_error_height) {
    auto param_shape_y = PartialShape{2, 480, 640, 1};
    auto param_shape_uv = PartialShape{2, 480, 320, 2};
    auto param_y = std::make_shared<op::v0::Parameter>(element::u8, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(element::u8, param_shape_uv);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param_y, param_uv), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_error_height_odd) {
    auto param_shape_y = PartialShape{2, 3, 2, 1};  // 3 is invalid, as UV shall be 2 times smaller
    auto param_shape_uv = PartialShape::dynamic();
    auto param_y = std::make_shared<op::v0::Parameter>(element::u8, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(element::u8, param_shape_uv);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param_y, param_uv), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_error_width) {
    auto param_shape_y = PartialShape{2, 480, 640, 1};
    auto param_shape_uv = PartialShape{2, 240, 640, 2};
    auto param_y = std::make_shared<op::v0::Parameter>(element::u8, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(element::u8, param_shape_uv);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param_y, param_uv), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_error_width_odd) {
    auto param_shape_y = PartialShape{2, 4, 3, 1};  // 3 is invalid, as UV width shall be 2 times smaller
    auto param_shape_uv = PartialShape::dynamic();
    auto param_y = std::make_shared<op::v0::Parameter>(element::u8, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(element::u8, param_shape_uv);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param_y, param_uv), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_error_channels) {
    auto param_shape_y = PartialShape{2, 480, 640, 1};
    auto param_shape_uv = PartialShape{2, 240, 320, 1};
    auto param_y = std::make_shared<op::v0::Parameter>(element::u8, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(element::u8, param_shape_uv);
    EXPECT_THROW(std::ignore = std::make_shared<TypeParam>(param_y, param_uv), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_error_many_types) {
    auto param_y = std::make_shared<op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param_u = std::make_shared<op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param_v = std::make_shared<op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto empty = std::make_shared<TypeParam>();
    empty->set_arguments(NodeVector{param_y, param_u, param_v});

    EXPECT_THROW(empty->constructor_validate_and_infer_types(), ov::AssertFailure);
}

TYPED_TEST_P(ConvertNV12BaseTest, shape_inference_2_plane_interval_dims_and_symbols) {
    auto param_shape_y = PartialShape{{2, 5}, {2, 20}, -1, 1};
    auto param_shape_uv = PartialShape{{2, 3}, {2, 12}, 2, -1};
    auto y_symbols = set_shape_symbols(param_shape_y);
    auto uv_symbols = set_shape_symbols(param_shape_uv);

    auto param_y = std::make_shared<op::v0::Parameter>(element::f32, param_shape_y);
    auto param_uv = std::make_shared<op::v0::Parameter>(element::f32, param_shape_uv);
    auto op = std::make_shared<TypeParam>(param_y, param_uv);

    EXPECT_EQ(op->output(0).get_partial_shape(), PartialShape({{2, 3}, {4, 20}, 4, 3}));
    EXPECT_EQ(op->output(0).get_element_type(), element::f32);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(y_symbols[0], y_symbols[1], y_symbols[2], nullptr));
}

REGISTER_TYPED_TEST_SUITE_P(ConvertNV12BaseTest,
                            shape_inference_default_ctor_single_plane,
                            shape_inference_single_tensor,
                            shape_inference_single_tensor_dynamic,
                            shape_inference_single_tensor_dynamic_dims,
                            shape_inference_single_tensor_dynamic_height,
                            shape_inference_single_tensor_dynamic_type,
                            shape_inference_single_tensor_error_channels,
                            shape_inference_single_tensor_error_dims_5,
                            shape_inference_single_tensor_error_dims_3,
                            shape_inference_single_tensor_error_height,
                            shape_inference_single_tensor_error_width_odd,
                            shape_inference_single_tensor_error_i8,
                            shape_inference_single_interval_dims_and_symbols,
                            shape_inference_default_ctor_2_planes,
                            shape_inference_2_plane_simple,
                            shape_inference_2_plane_dynamic,
                            shape_inference_2_plane_y_dynamic,
                            shape_inference_2_plane_uv_dynamic,
                            shape_inference_2_plane_dynamic_types,
                            shape_inference_2_plane_uv_type,
                            shape_inference_2_plane_error_type_mismatch,
                            shape_inference_2_plane_error_uv_type,
                            shape_inference_2_plane_error_5dims,
                            shape_inference_2_plane_error_3dims,
                            shape_inference_2_plane_error_batch,
                            shape_inference_2_plane_error_height,
                            shape_inference_2_plane_error_height_odd,
                            shape_inference_2_plane_error_width,
                            shape_inference_2_plane_error_width_odd,
                            shape_inference_2_plane_error_channels,
                            shape_inference_error_many_types,
                            shape_inference_2_plane_interval_dims_and_symbols);
