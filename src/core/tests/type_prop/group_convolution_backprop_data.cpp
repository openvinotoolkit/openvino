// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "convolution_shape_inference.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, group_convolution_backprop_data_shape_infer) {
    const PartialShape data_pshape{1, 16, 6, 6};       // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_pshape{2, 8, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t et = element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  Strides{},
                                                                  CoordinateDiff{},
                                                                  CoordinateDiff{},
                                                                  Strides{});

    EXPECT_EQ(gcbd->get_element_type(), element::f32);
    EXPECT_EQ(gcbd->get_output_shape(0), (Shape{1, 4, 8, 8}));
    EXPECT_EQ(gcbd->get_strides(), (Strides{1, 1}));
    EXPECT_EQ(gcbd->get_dilations(), (Strides{1, 1}));
    EXPECT_EQ(gcbd->get_pads_begin(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gcbd->get_pads_end(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gcbd->get_output_padding(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gcbd->get_auto_pad(), op::PadType::EXPLICIT);
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_with_output_shape_as_const) {
    const PartialShape data_pshape{1, 16, 5, 5};        // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_pshape{1, 16, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t et = element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(element::i64, Shape{2}, {3, 3});
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  output_shape,
                                                                  Strides{},
                                                                  Strides{},
                                                                  op::PadType::SAME_UPPER);

    ASSERT_EQ(gcbd->get_element_type(), element::f32);
    ASSERT_EQ(gcbd->get_output_shape(0), (Shape{1, 2, 3, 3}));
    ASSERT_EQ(gcbd->get_strides(), (Strides{1, 1}));
    ASSERT_EQ(gcbd->get_dilations(), (Strides{1, 1}));
    ASSERT_EQ(gcbd->get_pads_begin(), (CoordinateDiff{2, 2}));
    ASSERT_EQ(gcbd->get_pads_end(), (CoordinateDiff{2, 2}));
    ASSERT_EQ(gcbd->get_output_padding(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(gcbd->get_auto_pad(), op::PadType::SAME_UPPER);
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_with_output_shape_as_param) {
    const PartialShape data_pshape{1, 16, 5, 5};        // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_pshape{1, 16, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t et = element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2});
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  output_shape,
                                                                  Strides{},
                                                                  Strides{},
                                                                  op::PadType::SAME_UPPER);

    ASSERT_EQ(gcbd->get_element_type(), element::f32);
    ASSERT_EQ(gcbd->get_auto_pad(), op::PadType::SAME_UPPER);
    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{1, 2, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_with_output_shape_static_ranks_data_nc_dyn) {
    const PartialShape data_pshape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};  // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_pshape{1, 16, 2, 3, 3};                                 // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t et = element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(element::i64, Shape{2}, {3, 3});
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  output_shape,
                                                                  Strides{},
                                                                  Strides{},
                                                                  op::PadType::SAME_UPPER);

    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2, 3, 3}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_with_output_shape_static_ranks_filters_group_dyn) {
    const PartialShape data_pshape{Dimension::dynamic(), 16, 5, 5};        // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_pshape{Dimension::dynamic(), 16, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t et = element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(element::i64, Shape{2}, {3, 3});
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  output_shape,
                                                                  Strides{},
                                                                  Strides{},
                                                                  op::PadType::SAME_UPPER);

    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2, 3, 3}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_with_output_shape_static_ranks_filters_group_cin_dyn) {
    const PartialShape data_pshape{Dimension::dynamic(), 16, 5, 5};  // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_pshape{Dimension::dynamic(),
                                      Dimension::dynamic(),
                                      2,
                                      3,
                                      3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t et = element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(element::i64, Shape{2}, {3, 3});
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  output_shape,
                                                                  Strides{},
                                                                  Strides{},
                                                                  op::PadType::SAME_UPPER);

    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, 3}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_with_output_shape_static_ranks_data_cin_filters_group_dyn) {
    const PartialShape data_pshape{1, Dimension::dynamic(), 5, 5};         // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_pshape{Dimension::dynamic(), 16, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t et = element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(element::i64, Shape{2}, {3, 3});
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  output_shape,
                                                                  Strides{},
                                                                  Strides{},
                                                                  op::PadType::SAME_UPPER);

    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{1, Dimension::dynamic(), 3, 3}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_with_output_shape_static_ranks_filters_group_cout_dyn) {
    PartialShape data_pshape{Dimension::dynamic(), 16, 5, 5};                           // [N, C_IN * GROUPS, H, W]
    PartialShape filters_pshape{Dimension::dynamic(), 16, Dimension::dynamic(), 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    auto data_symbols = set_shape_symbols(data_pshape);
    auto filter_symbols = set_shape_symbols(filters_pshape);
    const element::Type_t et = element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(element::i64, Shape{2}, {3, 3});
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  output_shape,
                                                                  Strides{},
                                                                  Strides{},
                                                                  op::PadType::SAME_UPPER);

    EXPECT_THAT(get_shape_symbols(gcbd->get_output_partial_shape(0)),
                ElementsAre(data_symbols[0], filter_symbols[2], nullptr, nullptr));
    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, 3}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_static_ranks_data_nc_dyn) {
    const PartialShape data_pshape{Dimension::dynamic(), Dimension::dynamic(), 224, 224};  // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_pshape{4, 5, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t et = element::f32;

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  strides,
                                                                  padding_begin,
                                                                  padding_end,
                                                                  dilations);

    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 8, 447, 447}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_static_ranks_filters_group_dyn) {
    const PartialShape data_pshape{1, 20, 224, 224};                      // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_pshape{Dimension::dynamic(), 5, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t et = element::f32;

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  strides,
                                                                  padding_begin,
                                                                  padding_end,
                                                                  dilations);

    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{1, 8, 447, 447}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_static_ranks_filters_group_cin_dyn) {
    const PartialShape data_pshape{Dimension::dynamic(), 20, 224, 224};  // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_pshape{Dimension::dynamic(),
                                      Dimension::dynamic(),
                                      2,
                                      3,
                                      3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t et = element::f32;

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  strides,
                                                                  padding_begin,
                                                                  padding_end,
                                                                  dilations);

    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), Dimension::dynamic(), 447, 447}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_static_ranks_data_cin_filters_group_dyn) {
    const PartialShape data_pshape{1, Dimension::dynamic(), 224, 224};    // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_pshape{Dimension::dynamic(), 5, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t et = element::f32;

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  strides,
                                                                  padding_begin,
                                                                  padding_end,
                                                                  dilations);

    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{1, Dimension::dynamic(), 447, 447}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_static_ranks_filters_group_cout_dyn) {
    PartialShape data_pshape{1, 20, 224, 224};                                         // [N, C_IN * GROUPS, H, W]
    PartialShape filters_pshape{Dimension::dynamic(), Dimension::dynamic(), 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    auto symbols = set_shape_symbols(data_pshape);
    set_shape_symbols(filters_pshape);
    const element::Type_t et = element::f32;

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  strides,
                                                                  padding_begin,
                                                                  padding_end,
                                                                  dilations);
    EXPECT_THAT(get_shape_symbols(gcbd->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, nullptr, nullptr));
    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{1, Dimension::dynamic(), 447, 447}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_static_ranks_data_spatial_dim_dyn) {
    const PartialShape data_pshape{1, 20, 224, 224};                      // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_pshape{4, 5, 2, Dimension::dynamic(), 3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t et = element::f32;

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  strides,
                                                                  padding_begin,
                                                                  padding_end,
                                                                  dilations);

    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{1, 8, Dimension(445, -1), 447}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_static_ranks_filters_spatial_dim_dyn) {
    PartialShape data_pshape{Dimension::dynamic(), 20, 224, Dimension::dynamic()};  // [N, C_IN * GROUPS, H, W]
    PartialShape filters_pshape{4, 5, 2, 3, 3};                                     // [GROUPS, C_IN, C_OUT, kH, kW]
    auto symbols = set_shape_symbols(data_pshape);
    set_shape_symbols(filters_pshape);
    const element::Type_t et = element::f32;

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  strides,
                                                                  padding_begin,
                                                                  padding_end,
                                                                  dilations);

    EXPECT_THAT(get_shape_symbols(gcbd->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, nullptr, nullptr));
    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 8, 447, Dimension(1, -1)}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_with_output_shape_data_dyn) {
    const PartialShape data_pshape{PartialShape::dynamic()};
    const PartialShape filters_pshape{1, 16, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t et = element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(element::i64, Shape{2}, {3, 3});
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  output_shape,
                                                                  Strides{},
                                                                  Strides{},
                                                                  op::PadType::SAME_UPPER);

    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2, 3, 3}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_data_dyn) {
    const PartialShape data_pshape{PartialShape::dynamic()};
    const PartialShape filters_pshape{4, 5, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t et = element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  Strides{},
                                                                  CoordinateDiff{},
                                                                  CoordinateDiff{},
                                                                  Strides{});

    ASSERT_EQ(gcbd->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), 8, Dimension(3, -1), Dimension(3, -1)}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_with_output_shape_filters_dyn) {
    const PartialShape data_pshape{1, 16, Dimension::dynamic(), Dimension::dynamic()};  // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(element::i64, Shape{2}, {3, 3});
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  output_shape,
                                                                  Strides{},
                                                                  Strides{},
                                                                  op::PadType::SAME_UPPER);

    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{1, Dimension::dynamic(), 3, 3}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_filters_dyn) {
    const PartialShape data_pshape{1, 8, 224, 224};  // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  Strides{},
                                                                  CoordinateDiff{},
                                                                  CoordinateDiff{},
                                                                  Strides{});

    ASSERT_EQ(gcbd->get_output_partial_shape(0),
              (PartialShape{1, Dimension::dynamic(), Dimension(224, -1), Dimension(224, -1)}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_with_output_shape_as_const_data_and_filters_dyn) {
    const PartialShape data_pshape{PartialShape::dynamic()};
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = ov::op::v0::Constant::create(element::i64, Shape{3}, {3, 3, 3});
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  output_shape,
                                                                  Strides{},
                                                                  Strides{},
                                                                  op::PadType::SAME_UPPER);

    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, 3, 3}));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_with_output_shape_as_param_data_and_filters_dyn) {
    const PartialShape data_pshape{PartialShape::dynamic()};
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto output_shape = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  output_shape,
                                                                  Strides{},
                                                                  Strides{},
                                                                  op::PadType::SAME_UPPER);

    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape::dynamic(5)));
}

TEST(type_prop, group_convolution_backprop_data_shape_infer_data_and_filters_dyn) {
    const PartialShape data_pshape{PartialShape::dynamic()};
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f32;

    auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
    auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                  filters,
                                                                  Strides{},
                                                                  CoordinateDiff{},
                                                                  CoordinateDiff{},
                                                                  Strides{});

    ASSERT_EQ(gcbd->get_output_partial_shape(0), (PartialShape::dynamic()));
}

TEST(type_prop, group_convolution_backprop_data_invalid_et_inputs) {
    try {
        const PartialShape data_pshape{1, 16, 6, 6};       // [N, C_IN * GROUPS, H, W]
        const PartialShape filters_pshape{2, 8, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]

        const element::Type_t data_et = element::f16;
        const element::Type_t filters_et = element::f32;

        auto data = make_shared<ov::op::v0::Parameter>(data_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(filters_et, filters_pshape);
        auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                      filters,
                                                                      Strides{},
                                                                      CoordinateDiff{},
                                                                      CoordinateDiff{},
                                                                      Strides{});
        // data and filters should be of same element type
        FAIL() << "Incompatible element types not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Element types for data batch and filters do not match"));
    } catch (...) {
        FAIL() << "Element types validation check of inputs failed for unexpected reason";
    }

    try {
        const PartialShape data_pshape{1, 16, 6, 6};       // [N, C_IN * GROUPS, H, W]
        const PartialShape filters_pshape{2, 8, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]

        const element::Type boolean_et = element::boolean;

        auto data = make_shared<ov::op::v0::Parameter>(boolean_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(boolean_et, filters_pshape);
        auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                      filters,
                                                                      Strides{},
                                                                      CoordinateDiff{},
                                                                      CoordinateDiff{},
                                                                      Strides{});
        // data and filters must be of numeric element type
        FAIL() << "Boolean element type of inputs not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Element type of inputs must be numeric"));
    } catch (...) {
        FAIL() << "Numeric element types of data batch and filters validation check failed for "
                  "unexpected reason.";
    }

    try {
        const PartialShape data_pshape{1, 16, 5, 5};        // [N, C_IN * GROUPS, H, W]
        const PartialShape filters_pshape{1, 16, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]

        const element::Type_t inputs_et = element::f32;

        auto data = make_shared<ov::op::v0::Parameter>(inputs_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(inputs_et, filters_pshape);
        auto output_shape = ov::op::v0::Constant::create(inputs_et, Shape{2}, {3, 3});
        auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                      filters,
                                                                      output_shape,
                                                                      Strides{},
                                                                      Strides{},
                                                                      op::PadType::SAME_UPPER);
        // output shape input element type must be of integer type
        FAIL() << "Incompatible element types not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type for output shape should be of integer type");
    } catch (...) {
        FAIL() << "Element types validation check of inputs failed for unexpected reason";
    }
}

TEST(type_prop, group_convolution_backprop_data_invalid_input_ranks) {
    // data partial shape provided is rank 4 (Conv2D)
    // filter partial shape provided is rank 6 (Conv3D)
    try {
        const PartialShape data_pshape{1, 16, 6, 6};          // [N, C_IN * GROUPS, H, W]
        const PartialShape filters_pshape{2, 8, 2, 3, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW, kD]

        const element::Type_t inputs_et = element::f32;

        auto data = make_shared<ov::op::v0::Parameter>(inputs_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(inputs_et, filters_pshape);
        auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                      filters,
                                                                      Strides{},
                                                                      CoordinateDiff{},
                                                                      CoordinateDiff{},
                                                                      Strides{});
        // data and filters have incompatible ranks
        FAIL() << "Incompatible input ranks not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data and filters rank do not match"));
    } catch (...) {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }

    // data partial shape provided is rank 5 (Conv3D)
    // filter partial shape provided is rank 5 (Conv2D)
    try {
        const PartialShape data_pshape{1, 16, 6, 6, 6};    // [N, C_IN * GROUPS, H, W, D]
        const PartialShape filters_pshape{2, 8, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]

        const element::Type_t inputs_et = element::f32;

        auto data = make_shared<ov::op::v0::Parameter>(inputs_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(inputs_et, filters_pshape);
        auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                      filters,
                                                                      Strides{},
                                                                      CoordinateDiff{},
                                                                      CoordinateDiff{},
                                                                      Strides{});
        // data and weight have incompatible ranks
        FAIL() << "Incompatible input ranks not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data and filters rank do not match"));
    } catch (...) {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }

    try {
        const PartialShape data_pshape{1, 16, 5, 5};        // [N, C_IN * GROUPS, H, W]
        const PartialShape filters_pshape{1, 16, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]

        const element::Type_t inputs_et = element::f32;

        auto data = make_shared<ov::op::v0::Parameter>(inputs_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(inputs_et, filters_pshape);
        auto output_shape = ov::op::v0::Constant::create(element::i64, Shape{2, 1}, {3, 3});
        auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                      filters,
                                                                      output_shape,
                                                                      Strides{},
                                                                      Strides{},
                                                                      op::PadType::SAME_UPPER);
        // Output shape optional input must be of rank 1
        FAIL() << "Incompatible output shape input rank not detected.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input delivering output shape must have rank 1"));
    } catch (...) {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }
}

TEST(type_prop, group_convolution_backprop_data_invalid_input_channel_dims) {
    const Strides strides{1, 1};
    const Strides dilations{1, 1};
    const CoordinateDiff padding{2, 2};
    const element::Type_t inputs_et = element::f32;

    try {
        const PartialShape data_pshape{1, 16, 5, 5};          // [N, C_IN * GROUPS, H, W]
        const PartialShape filters_pshape{21, 16, 20, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]

        auto data = make_shared<ov::op::v0::Parameter>(inputs_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(inputs_et, filters_pshape);
        auto gcbd =
            make_shared<op::v1::GroupConvolutionBackpropData>(data, filters, strides, padding, padding, dilations);
        // data batch shape does not have correct dimension C_IN * GROUPS
        FAIL() << "Incompatibile input shapes not detected.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Input channels dimension of data batch is incompatible with filter groups or input channels."));
    } catch (...) {
        FAIL() << "Input shapes validation check failed for unexpected reason.";
    }

    try {
        const PartialShape data_pshape{1, 16, 5, 5};         // [N, C_IN * GROUPS, H, W]
        const PartialShape filters_pshape{4, 16, 20, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]

        auto data = make_shared<ov::op::v0::Parameter>(inputs_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(inputs_et, filters_pshape);
        auto gcbd =
            make_shared<op::v1::GroupConvolutionBackpropData>(data, filters, strides, padding, padding, dilations);
        // filter shape specifies GROUPS = 4 and C_IN = 16, while data batch shape specifies
        // dimension C_IN * GROUPS = 16
        FAIL() << "Incompatibile input shapes not detected.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Input channels dimension of data batch is incompatible with filter groups or input channels."));
    } catch (...) {
        FAIL() << "Input shapes validation check failed for unexpected reason.";
    }
}

TEST(type_prop, group_convolution_backprop_data_invalid_output_shape_spatial_dims) {
    const PartialShape data_pshape{1, 16, 5, 5};       // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_shape{1, 16, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t inputs_et = element::f32;

    try {
        auto data = make_shared<ov::op::v0::Parameter>(inputs_et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(inputs_et, filters_shape);
        auto output_shape = ov::op::v0::Constant::create(element::i64, Shape{3}, {3, 3, 3});
        auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                      filters,
                                                                      output_shape,
                                                                      Strides{},
                                                                      Strides{},
                                                                      op::PadType::SAME_UPPER);
        // output_shape has invalid spatials dimensions (should be 2)
        FAIL() << "Incompatible output shape optional input not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Output shape should be defined for all and only spatial dimensions."));
    } catch (...) {
        FAIL() << "Output shape validation check failed for unexpected reason.";
    }
}

TEST(type_prop, group_convolution_backprop_data_invalid_conv_param_spatial_dims) {
    const PartialShape data_pshape{1, 16, 6, 6};       // [N, C_IN * GROUPS, H, W]
    const PartialShape filters_pshape{2, 8, 2, 3, 3};  // [GROUPS, C_IN, C_OUT, kH, kW]
    const element::Type_t et = element::f32;

    // invalid strides spatial dimensions
    try {
        const Strides strides{1, 1, 1};
        const Strides dilations{1, 1};
        const CoordinateDiff pads_begin{0, 0};
        const CoordinateDiff pads_end{0, 0};

        auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(et, PartialShape::dynamic());
        auto gcbd =
            make_shared<op::v1::GroupConvolutionBackpropData>(data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid strides spatial dimensions not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Strides should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Strides spatial dimensions validation check failed for unexpected reason";
    }
    try {
        const Strides strides{1};
        const Strides dilations{1, 1};
        const CoordinateDiff pads_begin{0, 0};
        const CoordinateDiff pads_end{0, 0};

        auto data = make_shared<ov::op::v0::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
        auto gcbd =
            make_shared<op::v1::GroupConvolutionBackpropData>(data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid strides spatial dimensions not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Strides should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Strides spatial dimensions validation check failed for unexpected reason";
    }

    // invalid dilations spatial dimensions
    try {
        const Strides strides{1, 1};
        const Strides dilations{1};
        const CoordinateDiff pads_begin{0, 0};
        const CoordinateDiff pads_end{0, 0};

        auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(et, PartialShape::dynamic());
        auto gcbd =
            make_shared<op::v1::GroupConvolutionBackpropData>(data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid dilations spatial dimensions not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Dilations should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Dilations spatial dimensions validation check failed for unexpected reason";
    }
    try {
        const Strides strides{1, 1};
        const Strides dilations{1, 1, 1};
        const CoordinateDiff pads_begin{0, 0};
        const CoordinateDiff pads_end{0, 0};

        auto data = make_shared<ov::op::v0::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
        auto gcbd =
            make_shared<op::v1::GroupConvolutionBackpropData>(data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid dilations spatial dimensions not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Dilations spatial dimensions validation check failed for unexpected reason";
    }

    // invalid padding spatial dimensions
    try {
        const Strides strides{1, 1};
        const Strides dilations{1, 1};
        const CoordinateDiff pads_begin{0, 0, 0};
        const CoordinateDiff pads_end{0, 0};

        auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(et, PartialShape::dynamic());
        auto gcbd =
            make_shared<op::v1::GroupConvolutionBackpropData>(data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid padding spatial dimensions not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Pads begin and end should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Padding spatial dimensions validation check failed for unexpected reason";
    }
    try {
        const Strides strides{1, 1};
        const Strides dilations{1, 1};
        const CoordinateDiff pads_begin{0, 0};
        const CoordinateDiff pads_end{0};

        auto data = make_shared<ov::op::v0::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
        auto gcbd =
            make_shared<op::v1::GroupConvolutionBackpropData>(data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid padding spatial dimensions not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Pads begin and end should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Padding spatial dimensions validation check failed for unexpected reason";
    }

    // invalid output padding spatial dimensions
    try {
        const Strides strides{1, 1};
        const Strides dilations{1, 1};
        const CoordinateDiff pads_begin{0, 0};
        const CoordinateDiff pads_end{0, 0};
        const CoordinateDiff output_padding{0, 0, 0};
        const op::PadType auto_pad = op::PadType::EXPLICIT;

        auto data = make_shared<ov::op::v0::Parameter>(et, data_pshape);
        auto filters = make_shared<ov::op::v0::Parameter>(et, PartialShape::dynamic());
        auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                      filters,
                                                                      strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      dilations,
                                                                      auto_pad,
                                                                      output_padding);
        FAIL() << "Invalid output padding spatial dimensions not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Output padding spatial dimensions validation check failed for unexpected reason";
    }
    try {
        const Strides strides{1, 1};
        const Strides dilations{1, 1};
        const CoordinateDiff pads_begin{0, 0};
        const CoordinateDiff pads_end{0, 0};
        const CoordinateDiff output_padding{0};
        const op::PadType auto_pad = op::PadType::EXPLICIT;

        auto data = make_shared<ov::op::v0::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
        auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                      filters,
                                                                      strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      dilations,
                                                                      auto_pad,
                                                                      output_padding);
        FAIL() << "Invalid output padding spatial dimensions not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "should be defined for all and only spatial dimensions.");
    } catch (...) {
        FAIL() << "Output padding spatial dimensions validation check failed for unexpected reason";
    }
}

TEST(type_prop, group_convolution_backprop_data_default_constructed) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 1, 1, 3, 3, 3});
    const auto out_spatial = ov::op::v0::Constant::create(element::i32, Shape{3}, {5, 4, 10});

    const auto op = make_shared<op::v1::GroupConvolutionBackpropData>();
    op->set_arguments(OutputVector{data, filters, out_spatial});
    op->set_strides({1, 1, 1});
    op->set_dilations({1, 1, 1});
    op->set_pads_begin({2, 2, 2});
    op->set_pads_end({2, 2, 2});
    op->set_auto_pad(op::PadType::EXPLICIT);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_strides(), Strides({1, 1, 1}));
    EXPECT_EQ(op->get_dilations(), Strides({1, 1, 1}));
    EXPECT_EQ(op->get_pads_begin(), CoordinateDiff({2, 2, 2}));
    EXPECT_EQ(op->get_pads_end(), CoordinateDiff({2, 2, 2}));
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, 1, 5, 4, 10}));
}

TEST(type_prop, group_convolution_backprop_data_interval_shapes) {
    PartialShape data_batch_pshape{{1, 3}, {2, 6}, {1, 5}, {3, 10}, {20, 100}};
    PartialShape filters_pshape{{2, 3}, {1, 3}, 1, 3, 3, 3};
    PartialShape out_spatial_pshape{{2, 3}, -1, 10};
    auto data_symbols = set_shape_symbols(data_batch_pshape);
    auto filter_symbols = set_shape_symbols(filters_pshape);
    auto out_symbols = set_shape_symbols(out_spatial_pshape);

    const element::Type_t et = element::f32;
    const auto auto_pad = op::PadType::SAME_LOWER;

    const auto data_batch = make_shared<ov::op::v0::Parameter>(et, data_batch_pshape);
    const auto filters = make_shared<ov::op::v0::Parameter>(et, filters_pshape);
    const auto out_spatial_shape_of =
        make_shared<op::v0::ShapeOf>(make_shared<ov::op::v0::Parameter>(et, out_spatial_pshape));
    const auto op = make_shared<op::v1::GroupConvolutionBackpropData>(data_batch,
                                                                      filters,
                                                                      out_spatial_shape_of,
                                                                      Strides{},
                                                                      CoordinateDiff{},
                                                                      CoordinateDiff{},
                                                                      Strides{},
                                                                      auto_pad);

    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(data_symbols[0], filter_symbols[0], out_symbols[0], out_symbols[1], out_symbols[2]));
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{1, 3}, {2, 3}, {2, 3}, -1, 10}));
    EXPECT_EQ(op->get_pads_begin(), (CoordinateDiff{0, 0, 0}));
    EXPECT_EQ(op->get_pads_end(), (CoordinateDiff{0, 0, 0}));
}
