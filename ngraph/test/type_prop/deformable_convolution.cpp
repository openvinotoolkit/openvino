//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, deformable_conv_v1_partial_auto_padding_same)
{
    const PartialShape data_batch_shape{1, 4, 5, 5};
    const PartialShape deformable_shape{1, 4, 3, 3};
    const PartialShape filters_shape{4, 4, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;
    const int64_t group = 4;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto deformable_values = make_shared<op::Parameter>(element::f32, deformable_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                      deformable_values,
                                                                      filters,
                                                                      strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      dilations,
                                                                      auto_pad,
                                                                      group,
                                                                      deformable_group);

    ASSERT_TRUE(deformable_conv->get_output_partial_shape(0).same_scheme(PartialShape{1, 4, 5, 5}));
    ASSERT_EQ(deformable_conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(deformable_conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, deformable_conv_v1_partial_auto_padding_same_nc_dims_dynamic_same_lower)
{
    const PartialShape data_batch_shape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape deformable_shape{1, 4, 3, 3};
    const PartialShape filters_shape{4, 4, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;
    const int64_t group = 4;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto deformable_values = make_shared<op::Parameter>(element::f32, deformable_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                      deformable_values,
                                                                      filters,
                                                                      strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      dilations,
                                                                      auto_pad,
                                                                      group,
                                                                      deformable_group);

    ASSERT_TRUE(deformable_conv->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 4, 5, 5}));
    ASSERT_EQ(deformable_conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(deformable_conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, deformable_conv_v1_partial_auto_padding_same_nc_dims_dynamic_same_upper)
{
    const PartialShape data_batch_shape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape deformable_shape{1, 4, 2, 2};
    const PartialShape filters_shape{4, 4, 2, 2};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_UPPER;
    const int64_t group = 4;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto deformable_values = make_shared<op::Parameter>(element::f32, deformable_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                      deformable_values,
                                                                      filters,
                                                                      strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      dilations,
                                                                      auto_pad,
                                                                      group,
                                                                      deformable_group);

    ASSERT_TRUE(deformable_conv->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 4, 5, 5}));
    ASSERT_EQ(deformable_conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(deformable_conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, deformable_conv_v1_partial_auto_padding_same_spatial_dims_dynamic)
{
    const PartialShape data_batch_shape{1, 4, Dimension::dynamic(), 5};
    const PartialShape deformable_shape{1, 4, 3, 3};
    const PartialShape filters_shape{4, 4, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;
    const int64_t group = 4;
    const int64_t deformable_group = 2;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto deformable_values = make_shared<op::Parameter>(element::f32, deformable_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto deformable_conv = make_shared<op::v1::DeformableConvolution>(data_batch,
                                                                      deformable_values,
                                                                      filters,
                                                                      strides,
                                                                      pads_begin,
                                                                      pads_end,
                                                                      dilations,
                                                                      auto_pad,
                                                                      group,
                                                                      deformable_group);

    ASSERT_TRUE(deformable_conv->get_output_partial_shape(0).same_scheme(
        {1, 4, Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_EQ(deformable_conv->get_pads_begin(), (CoordinateDiff{}));
    ASSERT_EQ(deformable_conv->get_pads_end(), (CoordinateDiff{}));
}
