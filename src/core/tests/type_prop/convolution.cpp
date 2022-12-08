// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_shape_inference.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, conv_v1_partial_rank) {
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::v1::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 padding_below,
                                                 padding_above,
                                                 window_dilation_strides);

    ASSERT_TRUE(conv->get_output_partial_shape(0).is_dynamic());
}

TEST(type_prop, conv_v1_partial_auto_padding_same) {
    const PartialShape data_batch_shape{1, 1, 5, 5};
    const PartialShape filters_shape{1, 1, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv =
        make_shared<op::v1::Convolution>(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    ASSERT_EQ(conv->get_output_partial_shape(0), (PartialShape{1, 1, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, conv_v1_partial_auto_padding_same_nc_dims_dynamic_same_lower) {
    const PartialShape data_batch_shape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape filters_shape{1, 1, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv =
        make_shared<op::v1::Convolution>(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    ASSERT_EQ(conv->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 1, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, conv_v1_partial_auto_padding_same_nc_dims_dynamic_same_upper) {
    const PartialShape data_batch_shape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape filters_shape{1, 1, 2, 2};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv =
        make_shared<op::v1::Convolution>(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    ASSERT_EQ(conv->get_output_partial_shape(0), PartialShape({Dimension::dynamic(), 1, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, conv_v1_partial_auto_padding_same_spatial_dims_dynamic) {
    const PartialShape data_batch_shape{1, 1, Dimension::dynamic(), 5};
    const PartialShape filters_shape{1, 1, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv =
        make_shared<op::v1::Convolution>(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    ASSERT_EQ(conv->get_output_partial_shape(0), PartialShape({1, 1, Dimension::dynamic(), 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 1}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{0, 1}));
}

TEST(type_prop, conv_v1_partial_data_shape_dynamic) {
    const PartialShape data_batch_shape{PartialShape::dynamic()};
    const PartialShape filters_shape{1, 1, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv =
        make_shared<op::v1::Convolution>(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    ASSERT_EQ(conv->get_output_partial_shape(0),
              PartialShape({Dimension::dynamic(), 1, Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, convolution_default_constructed) {
    auto conv = make_shared<op::v1::Convolution>();
    conv->set_auto_pad(op::PadType::SAME_LOWER);

    const auto &input_shape = ov::PartialShape::dynamic(), filters_shape = ov::PartialShape{1, 1, 3, 3};
    const auto& input_shapes = std::vector<ov::PartialShape>{input_shape, filters_shape};
    std::vector<ov::PartialShape> output_shapes(1);
    auto pad_begin = CoordinateDiff{}, pad_end = CoordinateDiff{};

    int64_t num_spatial = calculate_num_spatial(conv.get(), input_shape, filters_shape, 2, 2);
    update_and_validate_attributes(conv.get(), num_spatial);
    EXPECT_NO_THROW(shape_infer(conv.get(), pad_begin, pad_end, input_shapes, output_shapes));
}