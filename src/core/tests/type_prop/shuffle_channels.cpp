// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/shuffle_channels.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, shuffle_channels_default_4D) {
    const auto data_input_shape = Shape{3, 9, 4, 5};
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_input_shape);
    const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data);

    EXPECT_EQ(shuffle_channels->get_element_type(), element::f32);
    EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
}

TEST(type_prop, shuffle_channels_basic_4D) {
    const auto data_input_shape = Shape{3, 9, 4, 5};
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_input_shape);
    const auto axis = 1;
    const auto group = 3;
    const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, axis, group);

    EXPECT_EQ(shuffle_channels->get_element_type(), element::f32);
    EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
}

TEST(type_prop, shuffle_channels_dynamic_4D) {
    auto data_input_shape = PartialShape{Dimension::dynamic(), Dimension(3, 9), 4, Dimension(4, 15)};
    auto symbols = set_shape_symbols(data_input_shape);
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_input_shape);
    const auto axis = 1;
    const auto group = 3;
    const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, axis, group);

    EXPECT_EQ(shuffle_channels->get_element_type(), element::f32);
    EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    EXPECT_THAT(get_shape_symbols(shuffle_channels->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, symbols[2], symbols[3]));
}

TEST(type_prop, shuffle_channels_dynamic_fully) {
    const auto data_input_shape = PartialShape::dynamic();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_input_shape);
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
        const auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_input_shape);
        const auto axis = 2;
        const auto group = 3;
        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, axis, group);

        EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    }
    {
        // 6D
        const auto data_input_shape = Shape{6, 2, 3, 9, 4, 5};
        const auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_input_shape);
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
        const auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_input_shape);
        const auto axis = 2;
        const auto group = 3;
        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, axis, group);

        EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    }
    {
        // 2D
        const auto data_input_shape = Shape{9, 20};
        const auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_input_shape);
        const auto axis = 0;
        const auto group = 3;
        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, axis, group);

        EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    }
    {
        // 1D
        const auto data_input_shape = Shape{9};
        const auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_input_shape);
        const auto axis = 0;
        const auto group = 3;
        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, axis, group);

        EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    }
}

TEST(type_prop, shuffle_channels_axis_validation) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f64, Shape{1, 2, 3, 4});

    OV_EXPECT_THROW(const auto op = make_shared<op::v0::ShuffleChannels>(data, -5, 5),
                    ov::AssertFailure,
                    HasSubstr("Axis -5 out of the tensor rank range [-4, 3]"));
}

TEST(type_prop, shuffle_channels_negative_axis_calculation) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f64, Shape{1, 2, 3, 4});

    const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, -3, 2);

    EXPECT_EQ(shuffle_channels->get_zero_based_axis(), 1);
}

TEST(type_prop, shuffle_channels_infer_shape_with_negative_axis_calculation) {
    // Only when the length of `axis` dimension is even, the shuffle_channels OP can work correctly.
    const auto group = 2;
    {
        const auto data_input_shape = Shape{1, 3, 5, 8};
        const auto data = make_shared<ov::op::v0::Parameter>(element::f64, data_input_shape);

        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, -1, group);
        EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    }
    {
        const auto data_input_shape = Shape{1, 3, 8, 5};
        const auto data = make_shared<ov::op::v0::Parameter>(element::f64, data_input_shape);

        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, -2, group);
        EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    }
    {
        const auto data_input_shape = Shape{8, 3, 5, 7};
        const auto data = make_shared<ov::op::v0::Parameter>(element::f64, data_input_shape);

        const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>(data, -4, group);
        EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_input_shape);
    }
}

TEST(type_prop, shuffle_channels_invalid_input_shape) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f64, Shape{});

    OV_EXPECT_THROW(const auto op = make_shared<op::v0::ShuffleChannels>(data, 0, 1),
                    NodeValidationFailure,
                    HasSubstr("The input tensor's shape is expected to be at least 1D."));
}

TEST(type_prop, shuffle_channels_invalid_groups_value) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f64, Shape{1, 2, 3, 15});

    OV_EXPECT_THROW(const auto op = make_shared<op::v0::ShuffleChannels>(data, -1, 2),
                    NodeValidationFailure,
                    HasSubstr("The channel dimension size has to be a multiple of the groups parameter value."));
}

TEST(type_prop, shuffle_channels_default_ctor) {
    const auto data_shape = PartialShape{{2, 5}, {0, 2}, 3, {2, -1}};
    const auto data = make_shared<ov::op::v0::Parameter>(element::i32, data_shape);

    const auto shuffle_channels = make_shared<op::v0::ShuffleChannels>();
    shuffle_channels->set_axis(-3);
    shuffle_channels->set_group(3);
    shuffle_channels->set_argument(0, data);
    shuffle_channels->validate_and_infer_types();

    EXPECT_EQ(shuffle_channels->get_axis(), -3);
    EXPECT_EQ(shuffle_channels->get_zero_based_axis(), 1);
    EXPECT_EQ(shuffle_channels->get_group(), 3);
    EXPECT_EQ(shuffle_channels->get_input_size(), 1);
    EXPECT_EQ(shuffle_channels->get_output_size(), 1);
    EXPECT_EQ(shuffle_channels->get_element_type(), element::i32);
    EXPECT_EQ(shuffle_channels->get_output_partial_shape(0), data_shape);
}
