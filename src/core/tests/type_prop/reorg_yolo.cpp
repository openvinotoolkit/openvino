// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset12.hpp"

using namespace ov;
using namespace ov::opset12;
using namespace testing;

class TypePropReorgYoloTest : public TypePropOpTest<ov::op::v0::ReorgYolo> {};

TEST_F(TypePropReorgYoloTest, default_ctor) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{1, 4, {13, 16}, 4});

    auto op = make_op();
    op->set_strides(2);
    op->set_arguments(OutputVector{data});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_strides(), Strides({2, 2}));
    EXPECT_EQ(op->get_input_size(), 1);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1, 16, {7, 8}, 2}));
}

TEST_F(TypePropReorgYoloTest, stride_2) {
    const auto in_shape = Shape{1, 64, 26, 26};
    size_t stride = 2;
    auto data_param = std::make_shared<Parameter>(element::f32, in_shape);
    auto reorg_yolo = make_op(data_param, stride);

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{1, 256, 13, 13};

    EXPECT_EQ(reorg_yolo->get_output_shape(0), expected_shape);
}

TEST_F(TypePropReorgYoloTest, stride_2_dynamic_shape) {
    const auto in_shape = PartialShape{-1, -1, -1, -1};
    size_t stride = 2;
    auto data_param = std::make_shared<Parameter>(element::f32, in_shape);
    auto reorg_yolo = make_op(data_param, stride);

    const auto expected_shape = PartialShape{-1, -1, -1, -1};

    EXPECT_EQ(reorg_yolo->get_output_partial_shape(0), expected_shape);
}

TEST_F(TypePropReorgYoloTest, stride_2_interval_shape_with_symbols) {
    auto in_shape = PartialShape{{1, 4}, {3, 9}, {16, 32}, {16, 32}};
    auto symbols = set_shape_symbols(in_shape);

    size_t stride = 2;
    auto data_param = std::make_shared<Parameter>(element::f32, in_shape);
    auto reorg_yolo = make_op(data_param, stride);

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    const auto expected_shape = PartialShape{{1, 4}, {12, 36}, {8, 16}, {8, 16}};

    EXPECT_EQ(reorg_yolo->get_output_partial_shape(0), expected_shape);
    EXPECT_THAT(get_shape_symbols(reorg_yolo->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, nullptr, nullptr));
}

TEST_F(TypePropReorgYoloTest, stride_2_batch_2) {
    const auto in_shape = Shape{2, 64, 26, 26};
    size_t stride = 2;
    auto data_param = std::make_shared<Parameter>(element::f32, in_shape);
    auto reorg_yolo = make_op(data_param, stride);

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{2, 256, 13, 13};

    EXPECT_EQ(reorg_yolo->get_output_shape(0), expected_shape);
}

TEST_F(TypePropReorgYoloTest, stride_2_smaller_H) {
    const auto in_shape = Shape{1, 24, 34, 62};
    size_t stride = 2;
    auto data_param = std::make_shared<Parameter>(element::f32, in_shape);
    auto reorg_yolo = make_op(data_param, stride);

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{1, 96, 17, 31};
    EXPECT_EQ(reorg_yolo->get_output_shape(0), expected_shape);
}

TEST_F(TypePropReorgYoloTest, stride_3) {
    const auto in_shape = Shape{1, 9, 3, 3};
    size_t stride = 3;
    auto data_param = std::make_shared<Parameter>(element::f32, in_shape);
    auto reorg_yolo = make_op(data_param, stride);

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape =
        Shape{in_shape[0], in_shape[1] * stride * stride, in_shape[2] / stride, in_shape[3] / stride};

    EXPECT_EQ(reorg_yolo->get_output_shape(0), expected_shape);
}

TEST_F(TypePropReorgYoloTest, catch_small_shape_stride) {
    const auto in_shape = Shape{1, 1, 4, 4};
    size_t stride = 2;
    auto data_param = std::make_shared<Parameter>(element::f32, in_shape);

    OV_EXPECT_THROW(std::ignore = make_op(data_param, stride), NodeValidationFailure, HasSubstr("stride"));
}

TEST_F(TypePropReorgYoloTest, data_shape_not_compatible_rank_4) {
    const auto data_param = std::make_shared<Parameter>(element::f32, PartialShape{1, 1, {13, 14}, 4});

    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3)),
                                          static_cast<size_t>(2)),
                    NodeValidationFailure,
                    HasSubstr("[N, C, H, W] input shape is required"));

    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<Parameter>(element::f32, PartialShape::dynamic(5)),
                                          static_cast<size_t>(2)),
                    NodeValidationFailure,
                    HasSubstr("[N, C, H, W] input shape is required"));
}

TEST_F(TypePropReorgYoloTest, h_dim_not_div_by_stride) {
    const auto data_param = std::make_shared<Parameter>(element::f32, PartialShape{1, 9, 13, 3});

    OV_EXPECT_THROW(std::ignore = make_op(data_param, static_cast<size_t>(3)),
                    NodeValidationFailure,
                    HasSubstr("H and W should be divisible by stride"));
}

TEST_F(TypePropReorgYoloTest, w_dim_not_div_by_stride) {
    const auto data_param = std::make_shared<Parameter>(element::f32, Shape{1, 9, 6, 4});

    OV_EXPECT_THROW(std::ignore = make_op(data_param, static_cast<size_t>(3)),
                    NodeValidationFailure,
                    HasSubstr("H and W should be divisible by stride"));
}

TEST_F(TypePropReorgYoloTest, channels_lower_than_sq_stride) {
    const auto data_param = std::make_shared<Parameter>(element::f32, PartialShape{1, 8, 6, 6});

    OV_EXPECT_THROW(std::ignore = make_op(data_param, static_cast<size_t>(3)),
                    NodeValidationFailure,
                    HasSubstr(" C >= (stride*stride) is required."));
}
