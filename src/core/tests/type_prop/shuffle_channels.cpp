// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, shuffle_channels_default_4D) {
    const auto data_input_shape = Shape{3, 9, 4, 5};
    const auto data = make_shared<op::Parameter>(element::f32, data_input_shape);
    const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data);

    EXPECT_EQ(shuffle_channels->get_element_type(), element::f32);
    EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
}

TEST(type_prop, shuffle_channels_basic_4D) {
    const auto data_input_shape = Shape{3, 9, 4, 5};
    const auto data = make_shared<op::Parameter>(element::f32, data_input_shape);
    const auto axis = 1;
    const auto group = 3;
    const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, axis, group);

    EXPECT_EQ(shuffle_channels->get_element_type(), element::f32);
    EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
}

TEST(type_prop, shuffle_channels_dynamic_4D) {
    const auto data_input_shape = PartialShape{Dimension::dynamic(), Dimension(3, 9), 4, Dimension(4, 15)};
    const auto data = make_shared<op::Parameter>(element::f32, data_input_shape);
    const auto axis = 1;
    const auto group = 3;
    const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, axis, group);

    EXPECT_EQ(shuffle_channels->get_element_type(), element::f32);
    EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
}

TEST(type_prop, shuffle_channels_dynamic_fully) {
    const auto data_input_shape = PartialShape::dynamic();
    const auto data = make_shared<op::Parameter>(element::f32, data_input_shape);
    const auto axis = 1;
    const auto group = 3;
    const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, axis, group);

    EXPECT_EQ(shuffle_channels->get_element_type(), element::f32);
    EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
}

TEST(type_prop, shuffle_channels_ND_bigger) {
    {
        // 5D
        const auto data_input_shape = Shape{2, 3, 9, 4, 5};
        const auto data = make_shared<op::Parameter>(element::f32, data_input_shape);
        const auto axis = 2;
        const auto group = 3;
        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, axis, group);

        EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    }
    {
        // 6D
        const auto data_input_shape = Shape{6, 2, 3, 9, 4, 5};
        const auto data = make_shared<op::Parameter>(element::f32, data_input_shape);
        const auto axis = 3;
        const auto group = 3;
        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, axis, group);

        EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    }
}

TEST(type_prop, shuffle_channels_ND_smaller) {
    {
        // 3D
        const auto data_input_shape = Shape{5, 4, 9};
        const auto data = make_shared<op::Parameter>(element::f32, data_input_shape);
        const auto axis = 2;
        const auto group = 3;
        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, axis, group);

        EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    }
    {
        // 2D
        const auto data_input_shape = Shape{9, 20};
        const auto data = make_shared<op::Parameter>(element::f32, data_input_shape);
        const auto axis = 0;
        const auto group = 3;
        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, axis, group);

        EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    }
    {
        // 1D
        const auto data_input_shape = Shape{9};
        const auto data = make_shared<op::Parameter>(element::f32, data_input_shape);
        const auto axis = 0;
        const auto group = 3;
        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, axis, group);

        EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    }
}

TEST(type_prop, shuffle_channels_axis_validation) {
    try {
        const auto data = make_shared<op::Parameter>(element::f64, Shape{1, 2, 3, 4});
        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, -5, 5);
        FAIL() << "ShuffleChannels validation did not work. Op node was created with incorrect "
                  "params.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The 'axis' parameter for ShuffleChannels has to point to one of the "
                             "input tensor's shape dimensions");
    }
}

TEST(type_prop, shuffle_channels_negative_axis_calculation) {
    const auto data = make_shared<op::Parameter>(element::f64, Shape{1, 2, 3, 4});

    const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, -3, 2);

    EXPECT_EQ(shuffle_channels->get_zero_based_axis(), 1);
}

TEST(type_prop, shuffle_channels_infer_shape_with_negative_axis_calculation) {
    // Only when the length of `axis` dimension is even, the shuffle_channels OP can work correctly.
    const auto group = 2;
    {
        const auto data_input_shape = Shape{1, 3, 5, 8};
        const auto data = make_shared<op::Parameter>(element::f64, data_input_shape);

        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, -1, group);
        EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    }
    {
        const auto data_input_shape = Shape{1, 3, 8, 5};
        const auto data = make_shared<op::Parameter>(element::f64, data_input_shape);

        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, -2, group);
        EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    }
    {
        const auto data_input_shape = Shape{8, 3, 5, 7};
        const auto data = make_shared<op::Parameter>(element::f64, data_input_shape);

        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, -4, group);
        EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    }
}

TEST(type_prop, shuffle_channels_invalid_input_shape) {
    try {
        const auto data = make_shared<op::Parameter>(element::f64, Shape{});
        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, 0, 1);
        FAIL() << "ShuffleChannels validation did not work. Op node was created with incorrect "
                  "params.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "The input tensor's shape is expected to be at least 1D.");
    }
}

TEST(type_prop, shuffle_channels_invalid_groups_value) {
    try {
        const auto data = make_shared<op::Parameter>(element::f64, Shape{1, 2, 3, 15});
        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, -1, 2);
        FAIL() << "ShuffleChannels validation did not work. Op node was created with incorrect "
                  "params.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The channel dimension size has to be a multiple of the groups parameter value.");
    }
}
