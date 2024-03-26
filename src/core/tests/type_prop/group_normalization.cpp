// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset12.hpp"

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

TEST(type_prop, group_normalization_symbols) {
    auto data_shape = PartialShape{1, 12, 6, 6};
    auto scale_shape = PartialShape{12};
    auto bias_shape = PartialShape{12};
    auto symbols = set_shape_symbols(data_shape);
    set_shape_symbols(scale_shape);
    set_shape_symbols(bias_shape);
    const auto data = std::make_shared<opset12::Parameter>(element::f32, data_shape);
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, scale_shape);
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, bias_shape);

    const auto gn = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 4, 0.00001f);
    EXPECT_THAT(get_shape_symbols(gn->get_output_partial_shape(0)), symbols);
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

TEST(type_prop, group_normalization_dynamic_ranks) {
    const auto data = std::make_shared<opset12::Parameter>(element::f16, PartialShape::dynamic());
    const auto scale = std::make_shared<opset12::Parameter>(element::f16, PartialShape::dynamic());
    const auto bias = std::make_shared<opset12::Parameter>(element::f16, PartialShape::dynamic());

    const auto gn = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 12, 0.00001f);
    EXPECT_EQ(gn->get_element_type(), element::f16);
    EXPECT_EQ(gn->get_output_partial_shape(0), (PartialShape::dynamic()));
}

TEST(type_prop, group_normalization_dynamic_intervals) {
    auto data_shape = PartialShape{2, Dimension{10, 20}, 6, 6};
    auto scale_shape = PartialShape{Dimension{10, 20}};
    auto bias_shape = PartialShape{Dimension{10, 20}};
    auto symbols = set_shape_symbols(data_shape);
    set_shape_symbols(scale_shape);
    set_shape_symbols(bias_shape);
    const auto data = std::make_shared<opset12::Parameter>(element::f32, data_shape);
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, scale_shape);
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, bias_shape);

    const auto gn = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 2, 0.00001f);
    EXPECT_EQ(gn->get_element_type(), element::f32);
    EXPECT_EQ(gn->get_output_partial_shape(0), (PartialShape{2, Dimension{10, 20}, 6, 6}));
    EXPECT_THAT(get_shape_symbols(gn->get_output_partial_shape(0)), symbols);
}

TEST(type_prop, group_normalization_incorrect_scale_shape) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, Shape{1, 12, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{13});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{12});

    OV_EXPECT_THROW(std::ignore = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 4, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The scale input shape needs to match the channel dimension in the data input"));
}

TEST(type_prop, group_normalization_incorrect_bias_shape) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, Shape{1, 12, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{12});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{14});

    OV_EXPECT_THROW(std::ignore = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 4, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The bias input shape needs to match the channel dimension in the data input"));
}

TEST(type_prop, group_normalization_incompatible_scale_and_bias) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, PartialShape{1, -1, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{2});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{4});

    OV_EXPECT_THROW(std::ignore = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 2, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The shapes of both scale and bias inputs need to match"));
}

TEST(type_prop, group_normalization_incorrect_scale_rank) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, Shape{1, 12, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{12, 12});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{12});

    OV_EXPECT_THROW(std::ignore = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 4, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The scale input is required to be 1D"));
}

TEST(type_prop, group_normalization_incorrect_bias_rank) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, Shape{1, 12, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{12});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{3, 14});

    OV_EXPECT_THROW(std::ignore = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 4, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The bias input is required to be 1D"));
}

TEST(type_prop, group_normalization_incompatible_channels_and_groups) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, PartialShape{1, 10, 6, 6});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{10});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{10});

    OV_EXPECT_THROW(std::ignore = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 3, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The number of channels is required to be evenly divisible by the number of groups"));
}

TEST(type_prop, group_normalization_incorrect_data_rank) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, PartialShape{10});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{1});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{1});

    OV_EXPECT_THROW(std::ignore = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 2, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The input tensor is required to be at least 2D"));
}

TEST(type_prop, group_normalization_negative_num_groups) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, PartialShape{1, 10});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{10});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{10});

    OV_EXPECT_THROW(std::ignore = std::make_shared<opset12::GroupNormalization>(data, scale, bias, -3, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The number of groups needs to be a positive integer value"));
}

TEST(type_prop, group_normalization_too_many_groups) {
    const auto data = std::make_shared<opset12::Parameter>(element::f32, PartialShape{1, 10});
    const auto scale = std::make_shared<opset12::Parameter>(element::f32, Shape{10});
    const auto bias = std::make_shared<opset12::Parameter>(element::f32, Shape{10});

    OV_EXPECT_THROW(std::ignore = std::make_shared<opset12::GroupNormalization>(data, scale, bias, 11, 0.00001f),
                    NodeValidationFailure,
                    HasSubstr("The number of groups must not exceed the number of channels in the input tensor"));
}
