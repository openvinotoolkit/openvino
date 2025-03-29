// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "convolution_shape_inference.hpp"

using namespace std;
using namespace testing;

// ---------------------------- v1 ----------------------------
TEST(type_prop, convolution_backprop_data_partial_auto_padding_upper) {
    const ov::Shape shape1{1, 512, 1, 37};
    const ov::Shape shape2{512, 256, 1, 1};
    const ov::Shape shape3{2};
    ov::Strides strides{1, 2};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};
    const auto auto_pad = ov::op::PadType::SAME_UPPER;

    auto in1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, shape1);
    auto in2 = make_shared<ov::op::v0::Parameter>(ov::element::f32, shape2);
    std::vector<int64_t> data = {1, 74};
    ov::element::Type type = ov::element::i64;
    auto in3 = make_shared<ov::op::v0::Constant>(type, shape3, data);

    auto conv = make_shared<ov::op::v1::ConvolutionBackpropData>(in1,
                                                                 in2,
                                                                 in3,
                                                                 strides,
                                                                 pads_begin,
                                                                 pads_end,
                                                                 dilations,
                                                                 auto_pad);
    conv->validate_and_infer_types();

    ASSERT_EQ(conv->get_pads_begin(), (ov::CoordinateDiff{0, 0}));
    ASSERT_EQ(conv->get_pads_end(), (ov::CoordinateDiff{0, 0}));
}

TEST(type_prop, convolution_backprop_data_partial_auto_padding_lower) {
    const ov::Shape shape1{1, 512, 1, 37};
    const ov::Shape shape2{512, 256, 1, 1};
    const ov::Shape shape3{2};
    ov::Strides strides{1, 2};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::Strides dilations{1, 1};
    const auto auto_pad = ov::op::PadType::SAME_LOWER;

    auto in1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, shape1);
    auto in2 = make_shared<ov::op::v0::Parameter>(ov::element::f32, shape2);
    std::vector<int64_t> data = {1, 74};
    ov::element::Type type = ov::element::i64;
    auto in3 = make_shared<ov::op::v0::Constant>(type, shape3, data);

    auto conv = make_shared<ov::op::v1::ConvolutionBackpropData>(in1,
                                                                 in2,
                                                                 in3,
                                                                 strides,
                                                                 pads_begin,
                                                                 pads_end,
                                                                 dilations,
                                                                 auto_pad);
    conv->validate_and_infer_types();

    ASSERT_EQ(conv->get_pads_begin(), (ov::CoordinateDiff{0, 0}));
    ASSERT_EQ(conv->get_pads_end(), (ov::CoordinateDiff{0, 0}));
}

TEST(type_prop, convolution_backprop_data_auto_pad_explicit_with_output_padding) {
    ov::PartialShape data_pshape{1, 16, 2, 2};
    ov::PartialShape filters_pshape{16, 6, 3, 3};
    auto d_symbols = set_shape_symbols(data_pshape);
    auto f_symbols = set_shape_symbols(filters_pshape);
    const ov::Strides strides{2, 2};
    const ov::Strides dilations{1, 1};
    const ov::CoordinateDiff padding_begin{1, 1};
    const ov::CoordinateDiff padding_end{1, 1};
    const ov::CoordinateDiff output_padding{1, 1};
    const ov::op::PadType auto_pad = ov::op::PadType::EXPLICIT;

    const ov::element::Type_t inputs_et = ov::element::f16;
    auto data = make_shared<ov::op::v0::Parameter>(inputs_et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(inputs_et, filters_pshape);
    auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                          filters,
                                                                          strides,
                                                                          padding_begin,
                                                                          padding_end,
                                                                          dilations,
                                                                          auto_pad,
                                                                          output_padding);

    EXPECT_THAT(get_shape_symbols(conv_backprop->get_output_partial_shape(0)),
                ElementsAre(d_symbols[0], f_symbols[1], nullptr, nullptr));
    ASSERT_EQ(conv_backprop->get_output_partial_shape(0), ov::PartialShape(ov::PartialShape{1, 6, 4, 4}));
    ASSERT_EQ(conv_backprop->get_pads_begin(), (ov::CoordinateDiff{1, 1}));
    ASSERT_EQ(conv_backprop->get_pads_end(), (ov::CoordinateDiff{1, 1}));
    ASSERT_EQ(conv_backprop->get_output_padding(), (ov::CoordinateDiff{1, 1}));
}

TEST(type_prop, convolution_backprop_data_auto_pad_same_with_output_padding_and_output_shape) {
    const ov::PartialShape data_pshape{1, 16, 2, 2};
    const ov::PartialShape filters_pshape{16, 6, 3, 3};

    const ov::Strides strides{2, 2};
    const ov::Strides dilations{1, 1};
    const ov::CoordinateDiff padding_begin{1, 1};
    const ov::CoordinateDiff padding_end{1, 1};
    const ov::CoordinateDiff output_padding{1, 1};
    const ov::op::PadType auto_pad = ov::op::PadType::SAME_LOWER;

    const ov::element::Type_t inputs_et = ov::element::f16;
    auto data = make_shared<ov::op::v0::Parameter>(inputs_et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(inputs_et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, 3});
    auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                          filters,
                                                                          output_shape,
                                                                          strides,
                                                                          padding_begin,
                                                                          padding_end,
                                                                          dilations,
                                                                          auto_pad,
                                                                          output_padding);

    EXPECT_EQ(conv_backprop->get_output_partial_shape(0), ov::PartialShape(ov::PartialShape{1, 6, 3, 3}));
    EXPECT_EQ(conv_backprop->get_pads_begin(), (ov::CoordinateDiff{1, 1}));
    EXPECT_EQ(conv_backprop->get_pads_end(), (ov::CoordinateDiff{2, 2}));
    EXPECT_EQ(conv_backprop->get_output_padding(), (ov::CoordinateDiff{1, 1}));
}

TEST(type_prop, convolution_backprop_data_output_shape_as_const) {
    const ov::PartialShape data_pshape{1, 16, 5, 5};
    const ov::PartialShape filters_pshape{16, 2, 3, 3};
    const ov::element::Type_t et = ov::element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, 3});
    auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                          filters,
                                                                          output_shape,
                                                                          ov::Strides{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::Strides{},
                                                                          ov::op::PadType::SAME_UPPER);

    EXPECT_EQ(conv_backprop->get_element_type(), ov::element::f32);
    EXPECT_EQ(conv_backprop->get_output_partial_shape(0), ov::PartialShape(ov::PartialShape{1, 2, 3, 3}));
    EXPECT_EQ(conv_backprop->get_strides(), (ov::Strides{1, 1}));
    EXPECT_EQ(conv_backprop->get_dilations(), (ov::Strides{1, 1}));
    EXPECT_EQ(conv_backprop->get_pads_begin(), (ov::CoordinateDiff{2, 2}));
    EXPECT_EQ(conv_backprop->get_pads_end(), (ov::CoordinateDiff{2, 2}));
    EXPECT_EQ(conv_backprop->get_output_padding(), (ov::CoordinateDiff{0, 0}));
    EXPECT_EQ(conv_backprop->get_auto_pad(), ov::op::PadType::SAME_UPPER);
}

TEST(type_prop, convolution_backprop_data_output_shape_as_param) {
    const ov::PartialShape data_pshape{1, 16, 5, 5};
    const ov::PartialShape filters_pshape{16, 2, 3, 3};
    const ov::element::Type_t et = ov::element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2});
    auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                          filters,
                                                                          output_shape,
                                                                          ov::Strides{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::Strides{},
                                                                          ov::op::PadType::SAME_UPPER);

    EXPECT_EQ(conv_backprop->get_element_type(), ov::element::f32);
    EXPECT_EQ(conv_backprop->get_auto_pad(), ov::op::PadType::SAME_UPPER);
    ASSERT_EQ(conv_backprop->get_output_partial_shape(0),
              ov::PartialShape(ov::PartialShape{1, 2, ov::Dimension::dynamic(), ov::Dimension::dynamic()}));
}

TEST(type_prop, convolution_backprop_data_with_output_shape_dyn_static_ranks_data_nc_dyn) {
    const ov::PartialShape data_pshape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 5, 5};
    const ov::PartialShape filters_pshape{16, 2, 3, 3};
    const ov::element::Type_t et = ov::element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, 3});
    auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                          filters,
                                                                          output_shape,
                                                                          ov::Strides{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::Strides{},
                                                                          ov::op::PadType::SAME_UPPER);

    ASSERT_EQ(conv_backprop->get_output_partial_shape(0),
              ov::PartialShape(ov::PartialShape{ov::Dimension::dynamic(), 2, 3, 3}));
}

TEST(type_prop, convolution_backprop_data_with_output_shape_dyn_static_ranks_filters_cin_dyn) {
    const ov::PartialShape data_pshape{ov::Dimension::dynamic(), 16, 5, 5};
    const ov::PartialShape filters_pshape{ov::Dimension::dynamic(), 6, 3, 3};
    const ov::element::Type_t et = ov::element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, 3});
    auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                          filters,
                                                                          output_shape,
                                                                          ov::Strides{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::Strides{},
                                                                          ov::op::PadType::SAME_UPPER);

    ASSERT_EQ(conv_backprop->get_output_partial_shape(0),
              ov::PartialShape(ov::PartialShape{ov::Dimension::dynamic(), 6, 3, 3}));
}

TEST(type_prop, convolution_backprop_data_with_output_shape_dyn_static_ranks_filters_cin_cout_dyn) {
    ov::PartialShape data_pshape{ov::Dimension::dynamic(), 16, 5, 5};
    ov::PartialShape filters_pshape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 3, 3};
    auto d_symbols = set_shape_symbols(data_pshape);
    auto f_symbols = set_shape_symbols(filters_pshape);
    const ov::element::Type_t et = ov::element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, 3});
    auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                          filters,
                                                                          output_shape,
                                                                          ov::Strides{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::Strides{},
                                                                          ov::op::PadType::SAME_UPPER);

    EXPECT_THAT(get_shape_symbols(conv_backprop->get_output_partial_shape(0)),
                ElementsAre(d_symbols[0], f_symbols[1], nullptr, nullptr));
    ASSERT_EQ(conv_backprop->get_output_partial_shape(0),
              ov::PartialShape(ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 3, 3}));
}

TEST(type_prop, convolution_backprop_data_dyn_static_ranks_data_nc_dyn) {
    const ov::PartialShape data_pshape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 224, 224};
    const ov::PartialShape filters_pshape{5, 2, 3, 3};
    const ov::element::Type_t et = ov::element::f32;

    const ov::Strides strides{2, 2};
    const ov::Strides dilations{1, 1};
    const ov::CoordinateDiff padding_begin{1, 1};
    const ov::CoordinateDiff padding_end{1, 1};

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto conv_backprop =
        make_shared<ov::op::v1::ConvolutionBackpropData>(data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_EQ(conv_backprop->get_output_partial_shape(0),
              ov::PartialShape(ov::PartialShape{ov::Dimension::dynamic(), 2, 447, 447}));
}

TEST(type_prop, convolution_backprop_data_dyn_static_ranks_filters_cin_dyn) {
    const ov::PartialShape data_pshape{ov::Dimension::dynamic(), 20, 224, 224};
    const ov::PartialShape filters_pshape{ov::Dimension::dynamic(), 2, 3, 3};
    const ov::element::Type_t et = ov::element::f32;

    const ov::Strides strides{2, 2};
    const ov::Strides dilations{1, 1};
    const ov::CoordinateDiff padding_begin{1, 1};
    const ov::CoordinateDiff padding_end{1, 1};

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto conv_backprop =
        make_shared<ov::op::v1::ConvolutionBackpropData>(data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_EQ(conv_backprop->get_output_partial_shape(0),
              ov::PartialShape(ov::PartialShape{ov::Dimension::dynamic(), 2, 447, 447}));
}

TEST(type_prop, convolution_backprop_data_dyn_static_ranks_filters_cin_cout_dyn) {
    const ov::PartialShape data_pshape{ov::Dimension::dynamic(), 20, 224, 224};
    const ov::PartialShape filters_pshape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 3, 3};
    const ov::element::Type_t et = ov::element::f32;

    const ov::Strides strides{2, 2};
    const ov::Strides dilations{1, 1};
    const ov::CoordinateDiff padding_begin{1, 1};
    const ov::CoordinateDiff padding_end{1, 1};

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto conv_backprop =
        make_shared<ov::op::v1::ConvolutionBackpropData>(data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_EQ(conv_backprop->get_output_partial_shape(0),
              ov::PartialShape(ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 447, 447}));
}

TEST(type_prop, convolution_backprop_data_dyn_static_ranks_data_spatial_dims_dyn) {
    const ov::PartialShape data_pshape{ov::Dimension::dynamic(), 4, ov::Dimension::dynamic(), 224};
    const ov::PartialShape filters_pshape{4, 16, 3, 3};
    const ov::element::Type_t et = ov::element::f32;

    const ov::Strides strides{2, 2};
    const ov::Strides dilations{1, 1};
    const ov::CoordinateDiff padding_begin{1, 1};
    const ov::CoordinateDiff padding_end{1, 1};

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto conv_backprop =
        make_shared<ov::op::v1::ConvolutionBackpropData>(data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_EQ(conv_backprop->get_output_partial_shape(0),
              ov::PartialShape(ov::PartialShape{ov::Dimension::dynamic(), 16, ov::Dimension(1, -1), 447}));
}

TEST(type_prop, convolution_backprop_data_dyn_static_ranks_filters_spatial_dims_dyn) {
    const ov::PartialShape data_pshape{ov::Dimension::dynamic(), 4, 224, 224};
    const ov::PartialShape filters_pshape{4, 16, 3, ov::Dimension::dynamic()};
    const ov::element::Type_t et = ov::element::f32;

    const ov::Strides strides{2, 2};
    const ov::Strides dilations{1, 1};
    const ov::CoordinateDiff padding_begin{1, 1};
    const ov::CoordinateDiff padding_end{1, 1};

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto conv_backprop =
        make_shared<ov::op::v1::ConvolutionBackpropData>(data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_EQ(conv_backprop->get_output_partial_shape(0),
              ov::PartialShape(ov::PartialShape{ov::Dimension::dynamic(), 16, 447, ov::Dimension(445, -1)}));
}

TEST(type_prop, convolution_backprop_data_with_output_shape_dyn_data_batch) {
    const ov::PartialShape data_pshape{ov::PartialShape::dynamic()};
    const ov::PartialShape filters_pshape{16, 2, 3, 3};
    const ov::element::Type_t et = ov::element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, 3});
    auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                          filters,
                                                                          output_shape,
                                                                          ov::Strides{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::Strides{});

    ASSERT_EQ(conv_backprop->get_output_partial_shape(0),
              ov::PartialShape(ov::PartialShape{ov::Dimension::dynamic(), 2, 3, 3}));
}

TEST(type_prop, convolution_backprop_data_with_output_shape_dyn_filters) {
    const ov::PartialShape data_pshape{1, 16, ov::Dimension::dynamic(), ov::Dimension::dynamic()};
    const ov::PartialShape filters_pshape{ov::PartialShape::dynamic()};
    const ov::element::Type_t et = ov::element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, 3});
    auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                          filters,
                                                                          output_shape,
                                                                          ov::Strides{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::Strides{});

    ASSERT_EQ(conv_backprop->get_output_partial_shape(0),
              ov::PartialShape(ov::PartialShape{1, ov::Dimension::dynamic(), 3, 3}));
}

TEST(type_prop, convolution_backprop_data_with_output_shape_as_const_dyn_data_and_filters) {
    const ov::PartialShape data_pshape{ov::PartialShape::dynamic()};
    const ov::PartialShape filters_pshape{ov::PartialShape::dynamic()};
    const ov::element::Type_t et = ov::element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {3, 3, 3});
    auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                          filters,
                                                                          output_shape,
                                                                          ov::Strides{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::Strides{});

    ASSERT_EQ(conv_backprop->get_output_partial_shape(0),
              ov::PartialShape(ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), 3, 3, 3}));
}

TEST(type_prop, convolution_backprop_data_with_output_shape_as_param_dyn_data_and_filters) {
    const ov::PartialShape data_pshape{ov::PartialShape::dynamic()};
    const ov::PartialShape filters_pshape{ov::PartialShape::dynamic()};
    const ov::element::Type_t et = ov::element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{3});
    auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                          filters,
                                                                          output_shape,
                                                                          ov::Strides{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::Strides{});

    ASSERT_EQ(conv_backprop->get_output_partial_shape(0), ov::PartialShape(ov::PartialShape::dynamic(5)));
}

TEST(type_prop, convolution_backprop_data_shape_dyn_data) {
    const ov::PartialShape data_pshape{ov::PartialShape::dynamic()};
    const ov::PartialShape filters_pshape{4, 2, 3, 3};
    const ov::element::Type_t et = ov::element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                          filters,
                                                                          ov::Strides{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::Strides{});

    ASSERT_EQ(
        conv_backprop->get_output_partial_shape(0),
        ov::PartialShape(ov::PartialShape{ov::Dimension::dynamic(), 2, ov::Dimension(3, -1), ov::Dimension(3, -1)}));
}

TEST(type_prop, convolution_backprop_data_shape_dyn_filters) {
    const ov::PartialShape data_pshape{1, 4, 224, 224};  // [N, C_IN, H, W]
    const ov::PartialShape filters_pshape{ov::PartialShape::dynamic()};
    const ov::element::Type_t et = ov::element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                          filters,
                                                                          ov::Strides{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::Strides{});

    ASSERT_EQ(conv_backprop->get_output_partial_shape(0),
              ov::PartialShape(
                  ov::PartialShape{1, ov::Dimension::dynamic(), ov::Dimension(224, -1), ov::Dimension(224, -1)}));
}

TEST(type_prop, convolution_backprop_data_dyn_data_and_filters) {
    const ov::PartialShape data_pshape{ov::PartialShape::dynamic()};
    const ov::PartialShape filters_pshape{ov::PartialShape::dynamic()};
    const ov::element::Type_t et = ov::element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                          filters,
                                                                          ov::Strides{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::CoordinateDiff{},
                                                                          ov::Strides{});

    ASSERT_EQ(conv_backprop->get_output_partial_shape(0), ov::PartialShape(ov::PartialShape::dynamic()));
}

TEST(type_prop, convolution_backprop_data_invalid_et_inputs) {
    const ov::PartialShape data_pshape{1, 16, 5, 5};
    const ov::PartialShape filters_pshape{16, 6, 3, 3};

    const ov::Strides strides{1, 1};
    const ov::Strides dilations{1, 1};
    const ov::CoordinateDiff padding_begin{1, 1};
    const ov::CoordinateDiff padding_end{1, 1};

    try {
        const ov::element::Type_t data_et = ov::element::f16;
        const ov::element::Type_t filters_et = ov::element::i64;

        auto data = make_shared<ov::op::v0::Parameter>(data_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(filters_et, filters_pshape);
        auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                              filters,
                                                                              strides,
                                                                              padding_begin,
                                                                              padding_end,
                                                                              dilations);
        FAIL() << "Invalid element type of inputs not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Element types for data batch and filters do not match");
    } catch (...) {
        FAIL() << "Element types of data batch and filters validation check failed for unexpected "
                  "reason.";
    }

    try {
        const ov::element::Type_t input_et = ov::element::boolean;

        auto data = make_shared<ov::op::v0::Parameter>(input_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(input_et, filters_pshape);
        auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                              filters,
                                                                              strides,
                                                                              padding_begin,
                                                                              padding_end,
                                                                              dilations);
        FAIL() << "Invalid element type of inputs not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type of inputs must be numeric");
    } catch (...) {
        FAIL() << "Numeric element types of data batch and filters validation check failed for "
                  "unexpected reason.";
    }

    try {
        const ov::element::Type_t et = ov::element::f32;

        auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
        auto output_shape = ov::op::v0::Constant::create(et, ov::Shape{2}, {3, 3});
        auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                              filters,
                                                                              output_shape,
                                                                              ov::Strides{},
                                                                              ov::CoordinateDiff{},
                                                                              ov::CoordinateDiff{},
                                                                              ov::Strides{},
                                                                              ov::op::PadType::SAME_UPPER);
        // output shape input element type must be of integer type
        FAIL() << "Invalid element type of output_shape input not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type for output shape should be of integer type");
    } catch (...) {
        FAIL() << "Element type of output_shape input validation check failed for unexpected reason";
    }
}

TEST(type_prop, convolution_backprop_data_invalid_input_ranks) {
    const ov::element::Type_t input_et = ov::element::f32;

    // data and filters don't have same rank
    try {
        const ov::PartialShape data_pshape{1, 20, 224, 224, 224};
        const ov::PartialShape filters_pshape{20, 10, 3, 3};

        auto data = make_shared<ov::op::v0::Parameter>(input_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(input_et, filters_pshape);
        auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                              filters,
                                                                              ov::Strides{},
                                                                              ov::CoordinateDiff{},
                                                                              ov::CoordinateDiff{},
                                                                              ov::Strides{});
        FAIL() << "Incompatible input ranks not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Data batch and filters rank do not match");
    } catch (...) {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }

    // data and filters don't have spatial dimensions
    try {
        const ov::PartialShape data_pshape{1, 20};
        const ov::PartialShape filters_pshape{20, 10};

        auto data = make_shared<ov::op::v0::Parameter>(input_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(input_et, filters_pshape);
        auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                              filters,
                                                                              ov::Strides{},
                                                                              ov::CoordinateDiff{},
                                                                              ov::CoordinateDiff{},
                                                                              ov::Strides{});
        FAIL() << "Incompatible input ranks not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a 3D, 4D or 5D tensor for the input. Got:");
    } catch (...) {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }

    // data and filters have 4 spatial dimensions (not supported)
    try {
        const ov::PartialShape data_pshape{1, 20, 224, 224, 224, 224};
        const ov::PartialShape filters_pshape{20, 10, 3, 3, 3, 3};

        auto data = make_shared<ov::op::v0::Parameter>(input_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(input_et, filters_pshape);
        auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                              filters,
                                                                              ov::Strides{},
                                                                              ov::CoordinateDiff{},
                                                                              ov::CoordinateDiff{},
                                                                              ov::Strides{});
        FAIL() << "Incompatible input ranks not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Expected a 3D, 4D or 5D tensor for the input. Got:");
    } catch (...) {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }

    try {
        const ov::PartialShape data_pshape{1, 16, 5, 5};
        const ov::PartialShape filters_shape{16, 2, 3, 3};

        auto data = make_shared<ov::op::v0::Parameter>(input_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(input_et, filters_shape);
        auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3, 1}, {3, 3, 3});
        auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                              filters,
                                                                              output_shape,
                                                                              ov::Strides{},
                                                                              ov::CoordinateDiff{},
                                                                              ov::CoordinateDiff{},
                                                                              ov::Strides{});
        // output_shape has rank 2, should be rank 1
        FAIL() << "Incompatible rank of output shape optional input not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input delivering output shape must have rank 1"));
    } catch (...) {
        FAIL() << "Output shape rank validation check failed for unexpected reason.";
    }
}

TEST(type_prop, convolution_backprop_data_invalid_input_channel_dims) {
    const ov::PartialShape data_pshape{1, 32, 5, 5};
    const ov::PartialShape filters_pshape{16, 20, 3, 3};
    const ov::element::Type_t inputs_et = ov::element::f32;

    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff padding{2, 2};

    auto data = make_shared<ov::op::v0::Parameter>(inputs_et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(inputs_et, filters_pshape);
    try {
        auto conv_backprop =
            make_shared<ov::op::v1::ConvolutionBackpropData>(data, filters, strides, padding, padding, dilations);
        // data input shape does not have correct dimension C_IN
        FAIL() << "Incompatibile input shapes not detected.";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Data batch channel count (32) does not match filter input channel count (16)"));
    } catch (...) {
        FAIL() << "Input shapes validation check failed for unexpected reason.";
    }
}

TEST(type_prop, convolution_backprop_data_invalid_output_shape_spatial_dims) {
    const ov::PartialShape data_pshape{1, 16, 5, 5};
    const ov::PartialShape filters_shape{16, 2, 3, 3};
    const ov::element::Type_t inputs_et = ov::element::f32;

    try {
        auto data = make_shared<ov::op::v0::Parameter>(inputs_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(inputs_et, filters_shape);
        auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {3, 3, 3});
        auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                              filters,
                                                                              output_shape,
                                                                              ov::Strides{},
                                                                              ov::CoordinateDiff{},
                                                                              ov::CoordinateDiff{},
                                                                              ov::Strides{});
        // output_shape has invalid spatial dimensions (should be 2)
        FAIL() << "Incompatible output shape optional input not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Output shape should be defined for all and only spatial dimensions."));
    } catch (...) {
        FAIL() << "Output shape validation check failed for unexpected reason.";
    }
}

TEST(type_prop, convolution_backprop_data_invalid_conv_param_spatial_dims) {
    const ov::PartialShape data_pshape{1, 20, 224, 224};
    const ov::PartialShape filters_pshape{20, 10, 3, 3};
    const ov::element::Type_t et = ov::element::f32;

    // invalid strides spatial dimensions
    try {
        ov::Strides strides{1, 1, 1};
        ov::Strides dilations{1, 1};
        ov::CoordinateDiff pads_begin{0, 0};
        ov::CoordinateDiff pads_end{0, 0};

        auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(et, ov::PartialShape::dynamic());
        auto conv_backprop =
            make_shared<ov::op::v1::ConvolutionBackpropData>(data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid strides spatial dimensions not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Strides should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Strides spatial dimensions validation check failed for unexpected reason";
    }
    try {
        ov::Strides strides{1};
        ov::Strides dilations{1, 1};
        ov::CoordinateDiff pads_begin{0, 0};
        ov::CoordinateDiff pads_end{0, 0};

        auto data = make_shared<ov::op::v0::Parameter>(et, ov::PartialShape::dynamic());
        auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
        auto conv_backprop =
            make_shared<ov::op::v1::ConvolutionBackpropData>(data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid strides spatial dimensions not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Strides should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Strides spatial dimensions validation check failed for unexpected reason";
    }

    // invalid dilations spatial dimensions
    try {
        ov::Strides strides{1, 1};
        ov::Strides dilations{1};
        ov::CoordinateDiff pads_begin{0, 0};
        ov::CoordinateDiff pads_end{0, 0};

        auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(et, ov::PartialShape::dynamic());
        auto conv_backprop =
            make_shared<ov::op::v1::ConvolutionBackpropData>(data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid dilations spatial dimensions not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Dilations should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Dilations spatial dimensions validation check failed for unexpected reason";
    }
    try {
        ov::Strides strides{1, 1};
        ov::Strides dilations{1, 1, 1};
        ov::CoordinateDiff pads_begin{0, 0};
        ov::CoordinateDiff pads_end{0, 0};

        auto data = make_shared<ov::op::v0::Parameter>(et, ov::PartialShape::dynamic());
        auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
        auto conv_backprop =
            make_shared<ov::op::v1::ConvolutionBackpropData>(data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid dilations spatial dimensions not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Dilations should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Dilations spatial dimensions validation check failed for unexpected reason";
    }

    // invalid padding spatial dimensions
    try {
        ov::Strides strides{1, 1};
        ov::Strides dilations{1, 1};
        ov::CoordinateDiff pads_begin{0, 0, 0};
        ov::CoordinateDiff pads_end{0, 0};

        auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(et, ov::PartialShape::dynamic());
        auto conv_backprop =
            make_shared<ov::op::v1::ConvolutionBackpropData>(data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid padding spatial dimensions not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Pads begin and end should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Padding spatial dimensions validation check failed for unexpected reason";
    }
    try {
        ov::Strides strides{1, 1};
        ov::Strides dilations{1, 1};
        ov::CoordinateDiff pads_begin{0, 0};
        ov::CoordinateDiff pads_end{0};

        auto data = make_shared<ov::op::v0::Parameter>(et, ov::PartialShape::dynamic());
        auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
        auto conv_backprop =
            make_shared<ov::op::v1::ConvolutionBackpropData>(data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid padding spatial dimensions not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Pads begin and end should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Padding spatial dimensions validation check failed for unexpected reason";
    }

    // invalid output padding spatial dimensions
    try {
        ov::Strides strides{1, 1};
        ov::Strides dilations{1, 1};
        ov::CoordinateDiff pads_begin{0, 0};
        ov::CoordinateDiff pads_end{0, 0};
        ov::CoordinateDiff output_padding{0, 0, 0};
        ov::op::PadType auto_pad = ov::op::PadType::EXPLICIT;

        auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(et, ov::PartialShape::dynamic());
        auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                              filters,
                                                                              strides,
                                                                              pads_begin,
                                                                              pads_end,
                                                                              dilations,
                                                                              auto_pad,
                                                                              output_padding);
        FAIL() << "Invalid output padding spatial dimensions not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Output padding should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Output padding spatial dimensions validation check failed for unexpected reason";
    }
    try {
        ov::Strides strides{1, 1};
        ov::Strides dilations{1, 1};
        ov::CoordinateDiff pads_begin{0, 0};
        ov::CoordinateDiff pads_end{0, 0};
        ov::CoordinateDiff output_padding{0};
        ov::op::PadType auto_pad = ov::op::PadType::EXPLICIT;

        auto data = make_shared<ov::op::v0::Parameter>(et, ov::PartialShape::dynamic());
        auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
        auto conv_backprop = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                              filters,
                                                                              strides,
                                                                              pads_begin,
                                                                              pads_end,
                                                                              dilations,
                                                                              auto_pad,
                                                                              output_padding);
        FAIL() << "Invalid output padding spatial dimensions not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Output padding should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Output padding spatial dimensions validation check failed for unexpected reason";
    }
}

TEST(type_prop, convolution_back_prop_data_default_constructed) {
    const auto data = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto filters = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 1, 3, 3});
    const auto out_spatial = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {5, 4, 10});

    const auto op = make_shared<ov::op::v1::ConvolutionBackpropData>();
    op->set_arguments(ov::OutputVector{data, filters, out_spatial});
    op->set_strides({1, 1, 1});
    op->set_dilations({1, 1, 1});
    op->set_pads_begin({2, 2, 2});
    op->set_pads_end({2, 2, 2});
    op->set_auto_pad(ov::op::PadType::EXPLICIT);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_strides(), ov::Strides({1, 1, 1}));
    EXPECT_EQ(op->get_dilations(), ov::Strides({1, 1, 1}));
    EXPECT_EQ(op->get_pads_begin(), ov::CoordinateDiff({2, 2, 2}));
    EXPECT_EQ(op->get_pads_end(), ov::CoordinateDiff({2, 2, 2}));
    EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape({-1, 1, 5, 4, 10}));
}

TEST(type_prop, convolution_back_prop_data_interval_shapes_output_shape_as_shape_of) {
    ov::PartialShape data_pshape{{1, 3}, {2, 6}, {1, 5}, {3, 10}, {20, 100}};
    ov::PartialShape filters_pshape{{2, 3}, {1, 3}, 3, 3, 3};
    ov::PartialShape out_spatial_pshape{3, {2, 4}, 3};

    auto d_symbols = set_shape_symbols(data_pshape);
    auto f_symbols = set_shape_symbols(filters_pshape);
    auto output_symbols = set_shape_symbols(out_spatial_pshape);

    const ov::element::Type_t et = ov::element::f32;
    ov::Strides strides{1, 2, 1};
    ov::Strides dilations{1, 1, 1};
    ov::CoordinateDiff pads_begin{0, 2, 1};
    ov::CoordinateDiff pads_end{0, 0, 0};
    const auto auto_pad = ov::op::PadType::SAME_LOWER;

    auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto out_spatial = make_shared<ov::op::v0::Parameter>(ov::element::i32, out_spatial_pshape);
    auto spatial_shape_of = std::make_shared<ov::op::v0::ShapeOf>(out_spatial);

    const auto op = make_shared<ov::op::v1::ConvolutionBackpropData>(data_batch,
                                                                     filters,
                                                                     spatial_shape_of,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     auto_pad);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(d_symbols[0], f_symbols[1], output_symbols[0], output_symbols[1], output_symbols[2]));
    EXPECT_EQ(op->get_output_partial_shape(0), ov::PartialShape({{1, 3}, {1, 3}, 3, {2, 4}, 3}));
    EXPECT_EQ(op->get_pads_begin(), (ov::CoordinateDiff{0, 0, 0}));
    EXPECT_EQ(op->get_pads_end(), (ov::CoordinateDiff{0, 0, 0}));
}
