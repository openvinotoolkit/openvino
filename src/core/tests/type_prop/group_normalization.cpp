// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset12.hpp"
#include "util/type_prop.hpp"

using namespace ov;
using namespace testing;

TEST(type_prop, group_normalization_basic) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, Shape{1, 12, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{12});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{12});

    const auto gn = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 4, 0.00001f);
    EXPECT_EQ(gn->get_element_type(), element::f32);
    EXPECT_EQ(gn->get_shape(), (Shape{1, 12, 6, 6}));
}

TEST(type_prop, group_normalization_labels) {
    auto data_shape = PartialShape{1, 12, 6, 6};
    set_shape_labels(data_shape, 43);
    const auto data = std::make_shared<opset12::Parameter>(element::f32, data_shape);
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{12});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{12});

    const auto gn = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 4, 0.00001f);
    EXPECT_THAT(get_shape_labels(gn->get_output_partial_shape(0)), ElementsAre(43, 44, 45, 46));
}

TEST(type_prop, group_normalization_dynamic_channels) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, PartialShape{1, -1, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{12});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{12});

    const auto gn = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 2, 0.00001f);
    EXPECT_EQ(gn->get_element_type(), element::f32);
    EXPECT_EQ(gn->get_output_partial_shape(0), (PartialShape{1, -1, 6, 6}));
}

TEST(type_prop, group_normalization_dynamic_scale) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, PartialShape{1, 4, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, PartialShape{-1});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, PartialShape{4});

    const auto gn = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 2, 0.00001f);
    EXPECT_EQ(gn->get_element_type(), element::f32);
    EXPECT_EQ(gn->get_output_partial_shape(0), (PartialShape{1, 4, 6, 6}));
}

TEST(type_prop, group_normalization_dynamic_bias) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, PartialShape{1, 4, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, PartialShape{4});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, PartialShape{-1});

    const auto gn = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 2, 0.00001f);
    EXPECT_EQ(gn->get_element_type(), element::f32);
    EXPECT_EQ(gn->get_output_partial_shape(0), (PartialShape{1, 4, 6, 6}));
}

TEST(type_prop, group_normalization_dynamic_rank) {
    const auto data = std::make_shared<opset12::Parameter>(element::f16, PartialShape::dynamic());
    const auto scale = std::make_shared<opset12::Parameter>(element::f16, PartialShape{6});
    const auto bias = std::make_shared<opset12::Parameter>(element::f16, PartialShape{6});

    const auto gn = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 3, 0.00001f);
    EXPECT_EQ(gn->get_element_type(), element::f16);
    EXPECT_EQ(gn->get_output_partial_shape(0), (PartialShape::dynamic()));
}

TEST(type_prop, group_normalization_dynamic_everything) {
    const auto data = std::make_shared<opset12::Parameter>(element::f16, PartialShape{3, -1, 10, 10});
    const auto scale = std::make_shared<opset12::Parameter>(element::f16, PartialShape{-1});
    const auto bias = std::make_shared<opset12::Parameter>(element::f16, PartialShape{-1});

    const auto gn = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 7, 0.00001f);
    EXPECT_EQ(gn->get_element_type(), element::f16);
    EXPECT_EQ(gn->get_output_partial_shape(0), (PartialShape{3, -1, 10, 10}));
}

TEST(type_prop, group_normalization_dynamic_intervals) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, PartialShape{2, Dimension{10, 20}, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, PartialShape{Dimension{10, 20}});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, PartialShape{Dimension{10, 20}});

    const auto gn = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 2, 0.00001f);
    EXPECT_EQ(gn->get_element_type(), element::f32);
    EXPECT_EQ(gn->get_output_partial_shape(0), (PartialShape{2, Dimension{10, 20}, 6, 6}));
}

TEST(type_prop, group_normalization_incorrect_scale_shape) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, Shape{1, 12, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{13});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{12});

    OV_EXPECT_THROW(std::make_shared<opset12::GroupNormalization>(data, scale, bias, 4, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The scale input shape needs to match the channel dimension in the data input"));
}

TEST(type_prop, group_normalization_incorrect_bias_shape) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, Shape{1, 12, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{12});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{14});

    OV_EXPECT_THROW(std::make_shared<opset12::GroupNormalization>(data, scale, bias, 4, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The bias input shape needs to match the channel dimension in the data input"));
}

TEST(type_prop, group_normalization_incompatible_scale_and_bias) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, PartialShape{1, -1, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{2});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{4});

    OV_EXPECT_THROW(std::make_shared<opset12::GroupNormalization>(data, scale, bias, 2, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The shapes of both scale and bias inputs need to match"));
}

TEST(type_prop, group_normalization_incorrect_scale_rank) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, Shape{1, 12, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{12, 12});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{12});

    OV_EXPECT_THROW(std::make_shared<opset12::GroupNormalization>(data, scale, bias, 4, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The scale input is required to be 1D"));
}

TEST(type_prop, group_normalization_incorrect_bias_rank) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, Shape{1, 12, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{12});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{3, 14});

    OV_EXPECT_THROW(std::make_shared<opset12::GroupNormalization>(data, scale, bias, 4, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The bias input is required to be 1D"));
}

TEST(type_prop, group_normalization_incompatible_channels_and_groups) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, PartialShape{1, 10, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{10});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{10});

    OV_EXPECT_THROW(std::make_shared<opset12::GroupNormalization>(data, scale, bias, 3, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The number of channels is required to be evenly divisible by the number of groups"));
}

TEST(type_prop, group_normalization_incorrect_data_rank) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, PartialShape{10});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{1});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{1});

    OV_EXPECT_THROW(std::make_shared<opset12::GroupNormalization>(data, scale, bias, 2, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The input tensor is required to be at least 2D"));
}
