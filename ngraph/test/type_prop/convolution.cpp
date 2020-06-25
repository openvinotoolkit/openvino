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

TEST(type_prop, conv_1d_deduce)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto conv = make_shared<op::Convolution>(param0, param1);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 91}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 91}); // output delta
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         Strides{1},
                                                         Strides{1},
                                                         CoordinateDiff{0},
                                                         CoordinateDiff{0},
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_filters_deduce)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 91}); // output delta
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            Strides{1},
                                                            Strides{1},
                                                            CoordinateDiff{0},
                                                            CoordinateDiff{0},
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_deduce_padded)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{1};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::Convolution>(
        param0, param1, move_strides, dilation_strides, padding_below, padding_above);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 96}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 96}); // output delta
    auto move_strides = Strides{1};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         dilation_strides,
                                                         padding_below,
                                                         padding_above,
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_filters_deduce_padded)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 96}); // output delta
    auto move_strides = Strides{1};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            dilation_strides,
                                                            padding_below,
                                                            padding_above,
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_deduce_strided)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 46}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_strided)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 46}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         Strides{1},
                                                         CoordinateDiff{0},
                                                         CoordinateDiff{0},
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_filters_deduce_strided)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 46}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            Strides{1},
                                                            CoordinateDiff{0},
                                                            CoordinateDiff{0},
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_deduce_strided_padded)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{2};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::Convolution>(
        param0, param1, move_strides, dilation_strides, padding_below, padding_above);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 48}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_strided_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 48}); // output delta
    auto move_strides = Strides{2};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         dilation_strides,
                                                         padding_below,
                                                         padding_above,
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_filters_deduce_strided_padded)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 48}); // output delta
    auto move_strides = Strides{2};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            dilation_strides,
                                                            padding_below,
                                                            padding_above,
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_deduce_strided_small_uneven)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2});
    auto move_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 2}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_strided_small_uneven)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 5};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 2}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         Strides{1},
                                                         CoordinateDiff{0},
                                                         CoordinateDiff{0},
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_filters_deduce_strided_small_uneven)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 5};
    Shape filters_shape{128, 3, 2};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});   // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 2}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            Strides{1},
                                                            CoordinateDiff{0},
                                                            CoordinateDiff{0},
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_deduce_strided_small_even)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2});
    auto move_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 3}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_strided_small_even)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 6};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 3}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         Strides{1},
                                                         CoordinateDiff{0},
                                                         CoordinateDiff{0},
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_filters_deduce_strided_small_even)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 6};
    Shape filters_shape{128, 3, 2};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 6});   // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 3}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            Strides{1},
                                                            CoordinateDiff{0},
                                                            CoordinateDiff{0},
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_deduce_window_dilated)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 82}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_window_dilated)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 82}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         dilate_strides,
                                                         CoordinateDiff{0},
                                                         CoordinateDiff{0},
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_filters_deduce_window_dilated)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 82}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            dilate_strides,
                                                            CoordinateDiff{0},
                                                            CoordinateDiff{0},
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_deduce_window_dilated_padded)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::Convolution>(
        param0, param1, move_strides, dilate_strides, padding_below, padding_above);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 87}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{1});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_window_dilated_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 87}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         dilate_strides,
                                                         padding_below,
                                                         padding_above,
                                                         Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_filters_deduce_window_dilated_padded)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});  // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 87}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            dilate_strides,
                                                            padding_below,
                                                            padding_above,
                                                            Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_deduce_window_dilated_data_dilated_padded)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto data_dilate_strides = Strides{3};
    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             move_strides,
                                             dilate_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 285}));

    EXPECT_EQ(conv->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides(), Strides{3});

    EXPECT_EQ(conv->get_padding_below(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_data_batch_deduce_window_dilated_data_dilated_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});   // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 285}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto data_dilate_strides = Strides{3};
    auto conv = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         param0,
                                                         param1,
                                                         move_strides,
                                                         dilate_strides,
                                                         padding_below,
                                                         padding_above,
                                                         data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{3});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_1d_back_filters_deduce_window_dilated_data_dilated_padded)
{
    // Deduce type
    // Shape data_batch_shape{64, 3, 100};
    Shape filters_shape{128, 3, 10};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});   // data batch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 285}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto data_dilate_strides = Strides{3};
    auto conv = make_shared<op::ConvolutionBackpropFilters>(param0,
                                                            filters_shape,
                                                            param1,
                                                            move_strides,
                                                            dilate_strides,
                                                            padding_below,
                                                            padding_above,
                                                            data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), filters_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{3});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_2d_deduce)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto conv = make_shared<op::Convolution>(param0, param1);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 91, 131}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_2d_deduce_padded)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto move_strides = Strides{1, 1};
    auto dilate_strides = Strides{1, 1};
    auto padding_below = CoordinateDiff{2, 3};
    auto padding_above = CoordinateDiff{3, 4};
    auto conv = make_shared<op::Convolution>(
        param0, param1, move_strides, dilate_strides, padding_below, padding_above);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 96, 138}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{2, 3}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{3, 4}));
}

TEST(type_prop, conv_2d_deduce_padded_neg)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto move_strides = Strides{1, 1};
    auto dilate_strides = Strides{1, 1};
    auto padding_below = CoordinateDiff{2, -3};
    auto padding_above = CoordinateDiff{3, -4};
    auto conv = make_shared<op::Convolution>(
        param0, param1, move_strides, dilate_strides, padding_below, padding_above);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 96, 124}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{2, -3}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{3, -4}));
}

struct DeduceAutoPadTest
    : ::testing::TestWithParam<
          std::tuple<Shape, Shape, Strides, Strides, CoordinateDiff, CoordinateDiff>>
{
};

TEST_P(DeduceAutoPadTest, same_upper)
{
    auto image_shape = std::get<0>(GetParam());
    image_shape.insert(image_shape.begin(), {1, 1}); // Add {N, C}
    auto filter_shape = std::get<1>(GetParam());
    filter_shape.insert(filter_shape.begin(), {1, 1}); // Add {O, I}
    auto param0 = make_shared<op::Parameter>(element::f32, image_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filter_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             std::get<2>(GetParam()),
                                             std::get<3>(GetParam()),
                                             CoordinateDiff(),
                                             CoordinateDiff(),
                                             Strides(),
                                             op::PadType::SAME_UPPER);
    EXPECT_EQ(conv->get_padding_below(), std::get<4>(GetParam()));
    EXPECT_EQ(conv->get_padding_above(), std::get<5>(GetParam()));

    auto no_dilation = std::all_of(std::get<3>(GetParam()).begin(),
                                   std::get<3>(GetParam()).end(),
                                   [](size_t i) { return i <= 1; });
    if (no_dilation)
    {
        auto max_pool = make_shared<op::MaxPool>(param0,
                                                 std::get<1>(GetParam()),
                                                 std::get<2>(GetParam()),
                                                 Shape(),
                                                 Shape(),
                                                 op::PadType::SAME_UPPER);
        CoordinateDiff padding_below(max_pool->get_padding_below().begin(),
                                     max_pool->get_padding_below().end());
        CoordinateDiff padding_above(max_pool->get_padding_above().begin(),
                                     max_pool->get_padding_above().end());
        EXPECT_EQ(padding_below, std::get<4>(GetParam()));
        EXPECT_EQ(padding_above, std::get<5>(GetParam()));

        auto avg_pool = make_shared<op::AvgPool>(param0,
                                                 std::get<1>(GetParam()),
                                                 std::get<2>(GetParam()),
                                                 Shape(),
                                                 Shape(),
                                                 false,
                                                 op::PadType::SAME_UPPER);
        CoordinateDiff pad_below(avg_pool->get_padding_below().begin(),
                                 avg_pool->get_padding_below().end());
        CoordinateDiff pad_above(avg_pool->get_padding_above().begin(),
                                 avg_pool->get_padding_above().end());
        EXPECT_EQ(pad_below, std::get<4>(GetParam()));
        EXPECT_EQ(pad_above, std::get<5>(GetParam()));
    }
}

TEST_P(DeduceAutoPadTest, same_lower)
{
    auto image_shape = std::get<0>(GetParam());
    image_shape.insert(image_shape.begin(), {1, 1}); // Add {N, C}
    auto filter_shape = std::get<1>(GetParam());
    filter_shape.insert(filter_shape.begin(), {1, 1}); // Add {O, I}
    auto param0 = make_shared<op::Parameter>(element::f32, image_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filter_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             std::get<2>(GetParam()),
                                             std::get<3>(GetParam()),
                                             CoordinateDiff(),
                                             CoordinateDiff(),
                                             Strides(),
                                             op::PadType::SAME_LOWER);
    EXPECT_EQ(conv->get_padding_above(), std::get<4>(GetParam()));
    EXPECT_EQ(conv->get_padding_below(), std::get<5>(GetParam()));
}

INSTANTIATE_TEST_CASE_P(type_prop,
                        DeduceAutoPadTest,
                        ::testing::Values(std::make_tuple(Shape{5, 6},
                                                          Shape{3, 4},
                                                          Strides{2, 1},
                                                          Strides{1, 1},
                                                          CoordinateDiff{1, 1},
                                                          CoordinateDiff{1, 2}),
                                          std::make_tuple(Shape{3, 3},
                                                          Shape{2, 2},
                                                          Strides{1, 1},
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{1, 1}),
                                          std::make_tuple(Shape{28, 28},
                                                          Shape{3, 3},
                                                          Strides{2, 2},
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{1, 1}),
                                          std::make_tuple(Shape{100, 150},
                                                          Shape{10, 20},
                                                          Strides{1, 1},
                                                          Strides{1, 1},
                                                          CoordinateDiff{4, 9},
                                                          CoordinateDiff{5, 10}),
                                          std::make_tuple(Shape{2},
                                                          Shape{1},
                                                          Strides{3},
                                                          Strides{1},
                                                          CoordinateDiff{0},
                                                          CoordinateDiff{0}),
                                          std::make_tuple(Shape{10, 1},
                                                          Shape{4, 1},
                                                          Strides{1, 1},
                                                          Strides{2, 1},
                                                          CoordinateDiff{3, 0},
                                                          CoordinateDiff{3, 0}),
                                          std::make_tuple(Shape{10, 5, 6},
                                                          Shape{3, 3, 4},
                                                          Strides{1, 2, 1},
                                                          Strides{2, 1, 1},
                                                          CoordinateDiff{2, 1, 1},
                                                          CoordinateDiff{2, 1, 2})), );

TEST(type_prop, conv_2d_deduce_strided)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto move_strides = Strides{2, 3};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 46, 44}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_2d_deduce_strided_window_dilated)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto move_strides = Strides{2, 3};
    auto dilate_strides = Strides{3, 2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 37, 38}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_2d_deduce_strided_window_dilated_data_dilated)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto move_strides = Strides{2, 3};
    auto dilate_strides = Strides{3, 2};
    auto padding_below = CoordinateDiff{0, 0};
    auto padding_above = CoordinateDiff{0, 0};
    auto data_dilate_strides = Strides{2, 3};
    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             move_strides,
                                             dilate_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 86, 137}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{2, 3}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_2d_deduce_strided_window_dilated_small)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2, 3});
    auto move_strides = Strides{2, 3};
    auto dilate_strides = Strides{3, 2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 2, 2}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_3d_deduce_strided_window_dilated_small)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2, 3, 2});
    auto move_strides = Strides{2, 3, 4};
    auto dilate_strides = Strides{3, 2, 2};
    auto conv = make_shared<op::Convolution>(param0, param1, move_strides, dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 2, 2, 2}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2, 2}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0, 0}));
}

TEST(type_prop, conv_3d_deduce_strided_window_dilated_data_dilated_small)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2, 3, 2});
    auto move_strides = Strides{2, 3, 4};
    auto dilate_strides = Strides{3, 2, 2};
    auto padding_below = CoordinateDiff{0, 0, 0};
    auto padding_above = CoordinateDiff{0, 0, 0};
    auto data_dilate_strides = Strides{2, 3, 2};
    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             move_strides,
                                             dilate_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 5, 6, 5}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{3, 2, 2}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{2, 3, 2}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0, 0}));
}

TEST(type_prop, conv_invalid_element_type_mismatch)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{3, 3, 3, 3});
    auto param1 = make_shared<op::Parameter>(element::i32, Shape{3, 3, 2, 2});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with element type mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Element types for data batch and filters do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_0d_input)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 0D input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data batch and filters must have rank of at least 3 "
                                         "(one batch axis, one input-channel axis, "
                                         "and at least one spatial dimension)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_1d_input)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 1D input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data batch and filters must have rank of at least 3 "
                                         "(one batch axis, one input-channel axis, "
                                         "and at least one spatial dimension)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_2d_input)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 6});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 2D input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data batch and filters must have rank of at least 3 "
                                         "(one batch axis, one input-channel axis, "
                                         "and at least one spatial dimension)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_0_batch_size)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{0, 6, 1});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{0, 6, 1});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 batch size not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Batch size is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_0_input_channels)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 0, 1});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 0, 1});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 input channels not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Data batch channel count and/or filter input channel count is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_wrong_number_of_filter_dimensions_too_many)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 2, 3, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too many filter dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch and filters rank do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_wrong_number_of_filter_dimensions_too_few)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 2, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too few filter dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch and filters rank do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_0_output_channels)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{0, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 output channels not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Filter output channel count is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_input_channel_mismatch)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 3, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with channel count mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Data batch channel count (2) does not match filter input channel count (3)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_movement_stride_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{2, 3, 8});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong movement stride rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape "
                        "{6,2,10,10}, so data item rank is 2 and filters have shape {6,2,3,3}, so "
                        "filters spatial rank is 2), data dilation (Strides{1, 1}), padding below "
                        "(CoordinateDiff{0, 0}), padding above (CoordinateDiff{0, 0}), filter "
                        "strides (Strides{2, 3, 8}), and filter dilation (Strides{1, 1}) do not "
                        "match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_window_dilation_stride_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{2, 3}, Strides{2, 3, 8});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong window dilation stride rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape "
                        "{6,2,10,10}, so data item rank is 2 and filters have shape {6,2,3,3}, so "
                        "filters spatial rank is 2), data dilation (Strides{1, 1}), padding below "
                        "(CoordinateDiff{0, 0}), padding above (CoordinateDiff{0, 0}), filter "
                        "strides (Strides{2, 3}), and filter dilation (Strides{2, 3, 8}) do not "
                        "match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_data_dilation_stride_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{2, 3},
                                                 Strides{2, 3},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0},
                                                 Strides{2, 3, 8});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong data dilation stride rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape "
                        "{6,2,10,10}, so data item rank is 2 and filters have shape {6,2,3,3}, so "
                        "filters spatial rank is 2), data dilation (Strides{2, 3, 8}), padding "
                        "below (CoordinateDiff{0, 0}), padding above (CoordinateDiff{0, 0}), "
                        "filter strides (Strides{2, 3}), and filter dilation (Strides{2, 3}) do "
                        "not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_padding_below_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{2, 3},
                                                 Strides{1, 1},
                                                 CoordinateDiff{0, 0, 0},
                                                 CoordinateDiff{0, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong padding-below rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Ranks for data item shape/filters shape (data batch has shape "
                "{6,2,10,10}, so data item rank is 2 and filters have shape {6,2,3,3}, so "
                "filters spatial rank is 2), data dilation (Strides{1, 1}), padding below "
                "(CoordinateDiff{0, 0, 0}), padding above (CoordinateDiff{0, 0}), filter "
                "strides (Strides{2, 3}), and filter dilation (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_padding_above_rank)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{2, 3},
                                                 Strides{2, 3},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong padding-above rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Ranks for data item shape/filters shape (data batch has shape "
                "{6,2,10,10}, so data item rank is 2 and filters have shape {6,2,3,3}, so "
                "filters spatial rank is 2), data dilation (Strides{1, 1}), padding below "
                "(CoordinateDiff{0, 0}), padding above (CoordinateDiff{0, 0, 0}), filter "
                "strides (Strides{2, 3}), and filter dilation (Strides{2, 3}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_input_spatial_size_negative_after_padding)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{1, 1},
                                                 Strides{1, 1},
                                                 CoordinateDiff{-4, 0},
                                                 CoordinateDiff{-7, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with negative-length post-padding spatial axis not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data shape after padding and dilation has dimension less "
                                         "than 1 (dim: -1) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_input_spatial_size_zero_after_padding)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{1, 1},
                                                 Strides{1, 1},
                                                 CoordinateDiff{-4, 0},
                                                 CoordinateDiff{-6, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length post-padding spatial axis not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data shape after padding and dilation has dimension less "
                                         "than 1 (dim: 0) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_input_spatial_size_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 0, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length spatial axis not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data shape after padding and dilation has "
                                         "dimension less than 1 (dim: 0) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_window_size_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 0});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length window axis not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window after dilation has dimension less than 1 (dim: 0) at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_window_dilation_stride_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{2, 3}, Strides{2, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong 0-length window dilation stride axis not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window dilation (Strides{2, 0}) has zero dimension at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_data_dilation_stride_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 Strides{2, 3},
                                                 Strides{2, 3},
                                                 CoordinateDiff{0, 0},
                                                 CoordinateDiff{0, 0},
                                                 Strides{2, 0});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong 0-length data dilation stride axis not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Data dilation (Strides{2, 0}) has zero dimension at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_dilated_window_too_large)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 8, 8});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{1, 1}, Strides{4, 4});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with oversized dilated window not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Window after dilation has dimension (dim: 9) larger than "
                                         "the data shape after padding (dim: 8) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_invalid_movement_stride_0)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 2, 3, 3});
    try
    {
        auto conv = make_shared<op::Convolution>(param0, param1, Strides{0, 1});

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong 0-length movement stride axis not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window strides (Strides{0, 1}) has zero dimension at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_window_strides_rank_wrong)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Window stride rank mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape ?, so data "
                        "item rank is ? and filters have shape ?, so filters spatial rank is ?), "
                        "data dilation (Strides{1, 1}), padding below (CoordinateDiff{0, 0}), "
                        "padding above (CoordinateDiff{0, 0}), filter strides (Strides{1, 1, 1}), "
                        "and filter dilation (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_window_strides_dim_zero)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 0};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Window stride with dimension zero not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window strides (Strides{1, 0}) has zero dimension at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_window_dilation_rank_wrong)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Window dilation rank mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape ?, so data "
                        "item rank is ? and filters have shape ?, so filters spatial rank is ?), "
                        "data dilation (Strides{1, 1}), padding below (CoordinateDiff{0, 0}), "
                        "padding above (CoordinateDiff{0, 0}), filter strides (Strides{1, 1}), and "
                        "filter dilation (Strides{1, 1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_window_dilation_dim_zero)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 0};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Window dilation with dimension zero not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window dilation (Strides{1, 0}) has zero dimension at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_padding_below_rank_wrong)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Padding below rank mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape ?, so data "
                        "item rank is ? and filters have shape ?, so filters spatial rank is ?), "
                        "data dilation (Strides{1, 1}), padding below (CoordinateDiff{0, 0, 0}), "
                        "padding above (CoordinateDiff{0, 0}), filter strides (Strides{1, 1}), and "
                        "filter dilation (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_padding_above_rank_wrong)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Padding above rank mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape ?, so data "
                        "item rank is ? and filters have shape ?, so filters spatial rank is ?), "
                        "data dilation (Strides{1, 1}), padding below (CoordinateDiff{0, 0}), "
                        "padding above (CoordinateDiff{0, 0, 0}), filter strides (Strides{1, 1}), "
                        "and filter dilation (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_data_dilation_rank_wrong)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Data dilation rank mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape ?, so data "
                        "item rank is ? and filters have shape ?, so filters spatial rank is ?), "
                        "data dilation (Strides{1, 1, 1}), padding below (CoordinateDiff{0, 0}), "
                        "padding above (CoordinateDiff{0, 0}), filter strides (Strides{1, 1}), and "
                        "filter dilation (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_dynamic_data_dilation_dim_zero)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 0};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Data dilation with dimension zero not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Data dilation (Strides{1, 0}) has zero dimension at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_dynamic_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic(4)};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_dynamic_data_batch_rank_wrong)
{
    PartialShape data_batch_shape{PartialShape::dynamic(5)};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Data batch rank mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape/filters shape (data batch has shape "
                        "{?,?,?,?,?}, so data item rank is 3 and filters have shape ?, so filters "
                        "spatial rank is ?), data dilation (Strides{1, 1}), padding below "
                        "(CoordinateDiff{0, 0}), padding above (CoordinateDiff{0, 0}), filter "
                        "strides (Strides{1, 1}), and filter dilation (Strides{1, 1}) do not "
                        "match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_dynamic_batch_size_known_ok)
{
    PartialShape data_batch_shape{
        64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_dynamic_batch_size_known_zero)
{
    PartialShape data_batch_shape{
        0, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Zero batch size not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Batch size is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_dynamic_input_channel_count_known_ok)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_dynamic_input_channel_count_known_zero)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape filters_shape{PartialShape::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Zero input channel count not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Data batch channel count and/or filter input channel count is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_static_dynamic_output_channel_count_known_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic(4)};
    PartialShape filters_shape{
        32, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 32, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, conv_partial_rank_dynamic_rank_static_dynamic_output_channel_count_known_zero)
{
    PartialShape data_batch_shape{PartialShape::dynamic(4)};
    PartialShape filters_shape{0, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Zero output channel count not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Filter output channel count is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_dynamic_rank_static_dynamic_input_channel_count_known_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic(4)};
    PartialShape filters_shape{Dimension::dynamic(), 4, Dimension::dynamic(), Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, conv_partial_rank_dynamic_rank_static_dynamic_input_channel_count_known_zero)
{
    PartialShape data_batch_shape{PartialShape::dynamic(4)};
    PartialShape filters_shape{Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Zero input channel count not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Data batch channel count and/or filter input channel count is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_static_dynamic_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic(4)};
    PartialShape filters_shape{PartialShape::dynamic(4)};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_static_dynamic_arg_ranks_mismatch)
{
    PartialShape data_batch_shape{PartialShape::dynamic(5)};
    PartialShape filters_shape{PartialShape::dynamic(4)};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Argument rank mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data batch and filters rank do not match (data batch "
                                         "shape: {?,?,?,?,?}, filters shape: {?,?,?,?})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_static_dynamic_input_channel_counts_known_ok)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape filters_shape{Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_static_dynamic_input_channel_counts_mismatch)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape filters_shape{
        Dimension::dynamic(), 22, Dimension::dynamic(), Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Input channel count mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Data batch channel count (3) does not match filter input channel count (22)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_known_ok)
{
    PartialShape data_batch_shape{64, 3, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape filters_shape{100, 3, Dimension::dynamic(), Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 100, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop,
     conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_known_ok)
{
    PartialShape data_batch_shape{64, 3, 200, Dimension::dynamic()};
    PartialShape filters_shape{100, 3, 5, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 100, 196, Dimension::dynamic()}));
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_known_filters_too_big)
{
    PartialShape data_batch_shape{64, 3, 200, Dimension::dynamic()};
    PartialShape filters_shape{100, 3, 201, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Oversize filter not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Window after dilation has dimension (dim: 201) larger "
                                         "than the data shape after padding (dim: 200) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_known_filters_not_too_big_after_padding)
{
    PartialShape data_batch_shape{64, 3, 200, Dimension::dynamic()};
    PartialShape filters_shape{100, 3, 201, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{2, 0};
    CoordinateDiff padding_above{-1, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 100, 1, Dimension::dynamic()}));
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_known_filters_not_too_big_after_data_dilation)
{
    PartialShape data_batch_shape{64, 3, 200, Dimension::dynamic()};
    PartialShape filters_shape{100, 3, 201, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{2, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 100, 199, Dimension::dynamic()}));
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_known_filters_not_too_big_after_data_dilation_strided)
{
    PartialShape data_batch_shape{64, 3, 200, Dimension::dynamic()};
    PartialShape filters_shape{100, 3, 201, Dimension::dynamic()};
    Strides window_movement_strides{3, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{2, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 100, 67, Dimension::dynamic()}));
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_known_filters_too_big_after_filter_dilation)
{
    PartialShape data_batch_shape{64, 3, 200, Dimension::dynamic()};
    PartialShape filters_shape{100, 3, 101, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{2, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Oversize filter after window dilation not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Window after dilation has dimension (dim: 201) larger "
                                         "than the data shape after padding (dim: 200) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_zero_data_batch_dim)
{
    PartialShape data_batch_shape{64, 3, 200, 0};
    PartialShape filters_shape{100, 3, 5, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Zero dimension in data batch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data shape after padding and dilation has "
                                         "dimension less than 1 (dim: 0) at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_positive_data_batch_dim_after_padding)
{
    PartialShape data_batch_shape{64, 3, 200, 0};
    PartialShape filters_shape{100, 3, 5, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 2};
    CoordinateDiff padding_above{0, -1};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_EQ(conv->get_output_element_type(0), element::f32);
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 100, 196, Dimension::dynamic()}));
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_zero_data_batch_dim_after_padding)
{
    PartialShape data_batch_shape{64, 3, 200, 20};
    PartialShape filters_shape{100, 3, 5, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, 0};
    CoordinateDiff padding_above{0, -20};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Zero padded dimension in data batch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data shape after padding and dilation has "
                                         "dimension less than 1 (dim: 0) at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    conv_partial_rank_static_dynamic_rank_static_dynamic_all_nonspatial_some_spatial_negative_data_batch_dim_after_padding)
{
    PartialShape data_batch_shape{64, 3, 200, 20};
    PartialShape filters_shape{100, 3, 5, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{0, -1};
    CoordinateDiff padding_above{0, -20};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        auto conv = make_shared<op::Convolution>(param0,
                                                 param1,
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides);

        FAIL() << "Negative padded dimension in data batch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data shape after padding and dilation has dimension less "
                                         "than 1 (dim: -1) at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_partial_dynamic_et)
{
    // For this test the exact shape parameters are kind of arbitrary---just copied and pasted
    // from some known-"OK" test above. We're only concerned about the element types.
    PartialShape data_batch_shape{64, 3, 200, Dimension::dynamic()};
    PartialShape filters_shape{100, 3, 201, Dimension::dynamic()};
    Strides window_movement_strides{1, 1};
    Strides window_dilation_strides{1, 1};
    CoordinateDiff padding_below{2, 0};
    CoordinateDiff padding_above{-1, 0};
    Strides data_dilation_strides{1, 1};

    auto param0 = make_shared<op::Parameter>(element::dynamic, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::dynamic, filters_shape);

    auto conv = make_shared<op::Convolution>(param0,
                                             param1,
                                             window_movement_strides,
                                             window_dilation_strides,
                                             padding_below,
                                             padding_above,
                                             data_dilation_strides);

    ASSERT_TRUE(conv->get_output_element_type(0).is_dynamic());
    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 100, 1, Dimension::dynamic()}));
}

TEST(type_prop, conv_bprop_filter_v1_output_partial_shape_dynamic)
{
    Shape shape_data{64, 3, 100};
    auto data = make_shared<op::Parameter>(element::f32, shape_data);
    Shape shape_delta{64, 128, 96};
    auto deltas = make_shared<op::Parameter>(element::f32, shape_delta);
    auto filters_shape = make_shared<op::Parameter>(element::i64, Shape{128, 3, 10});
    auto strides = Strides{1};
    auto dilations = Strides{1};
    auto padding_begin = CoordinateDiff{2};
    auto padding_end = CoordinateDiff{3};
    auto conv1 = make_shared<op::v1::ConvolutionBackpropFilters>(
        data, deltas, filters_shape, strides, dilations, padding_begin, padding_end);

    ASSERT_TRUE(conv1->get_output_partial_shape(0).is_dynamic());
}

TEST(type_prop, conv_bprop_data_v1_output_partial_shape_dynamic)
{
    Shape shape_filter{6, 3, 3, 3};
    auto filters = make_shared<op::Parameter>(element::f32, shape_filter);
    Shape shape_delta{2, 6, 3, 3};
    auto deltas = make_shared<op::Parameter>(element::f32, shape_delta);
    Shape shape_data_batch_shape{2, 3, 5, 5};
    auto data_batch_shape = make_shared<op::Parameter>(element::i64, Shape{2, 3, 5, 5});
    auto strides = Strides{1, 1};
    auto dilations = Strides{1, 1};
    auto padding_begin = CoordinateDiff{0, 0};
    auto padding_end = CoordinateDiff{0, 0};

    auto conv1 = make_shared<op::v1::ConvolutionBackpropData>(
        deltas, filters, data_batch_shape, strides, padding_begin, padding_end, dilations);

    ASSERT_TRUE(conv1->get_output_partial_shape(0).is_dynamic());
}

TEST(type_prop, conv_bprop_data_v1_output_partial_shape_dynamic_static_rank)
{
    PartialShape shape_filter{20, 10, 3, 3};
    auto filters = make_shared<op::Parameter>(element::f32, shape_filter);
    PartialShape shape_delta{Dimension(), 20, 224, 224};
    auto deltas = make_shared<op::Parameter>(element::f32, shape_delta);
    auto strides = Strides{2, 2};
    auto dilations = Strides{1, 1};
    auto padding_begin = CoordinateDiff{1, 1};
    auto padding_end = CoordinateDiff{1, 1};

    auto conv1 = make_shared<op::v1::ConvolutionBackpropData>(
        deltas, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_TRUE(conv1->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(conv1->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(conv1->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv1->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 10, 447, 447}));
}

TEST(type_prop, conv_v1_partial_rank)
{
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

TEST(type_prop, conv_v1_partial_auto_padding_same)
{
    const PartialShape data_batch_shape{1, 1, 5, 5};
    const PartialShape filters_shape{1, 1, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::v1::Convolution>(data_batch,
                                                 filters,
                                                 strides,
                                                 pads_begin,
                                                 pads_end,
                                                 dilations,
                                                 auto_pad);

    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape{1, 1, 5, 5}));
}

TEST(type_prop, conv_v1_partial_auto_padding_same_nc_dims_dynamic)
{
    const PartialShape data_batch_shape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape filters_shape{1, 1, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::v1::Convolution>(data_batch,
                                                 filters,
                                                 strides,
                                                 pads_begin,
                                                 pads_end,
                                                 dilations,
                                                 auto_pad);

    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme({Dimension::dynamic(), 1, 5, 5}));
}

TEST(type_prop, conv_v1_partial_auto_padding_same_spatial_dims_dynamic)
{
    const PartialShape data_batch_shape{1, 1, Dimension::dynamic(), 5};
    const PartialShape filters_shape{1, 1, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::v1::Convolution>(data_batch,
                                                filters,
                                                strides,
                                                pads_begin,
                                                pads_end,
                                                dilations,
                                                auto_pad);

    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme({1, 1, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, deformable_conv_incorrect_group)
{
    const PartialShape data_batch_shape{1, 3, 96, 96};
    const PartialShape deformable_values_shape{1, 50, 5, 5};
    const PartialShape filters_shape{4, 3, 5, 5};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, deformable_values_shape);
    auto param2 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        make_shared<op::v1::DeformableConvolution>(param0,
                                                   param1,
                                                   param2,
                                                   Strides{},
                                                   CoordinateDiff{},
                                                   CoordinateDiff{},
                                                   Strides{},
                                                   op::PadType::EXPLICIT,
                                                   2);

        FAIL() << "DeformableConvolution created with incorrect 'group' value";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "input data shape must be evenly divisible");
    }

    try
    {
        make_shared<op::v1::DeformableConvolution>(param0,
                                                   param1,
                                                   param2,
                                                   Strides{},
                                                   CoordinateDiff{},
                                                   CoordinateDiff{},
                                                   Strides{},
                                                   op::PadType::EXPLICIT,
                                                   3);

        FAIL() << "DeformableConvolution created with incorrect 'group' value";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "weights shape must be evenly divisible");
    }
}

TEST(type_prop, deformable_conv_incorrect_deformable_group)
{
    const PartialShape data_batch_shape{1, 3, 96, 96};
    const PartialShape deformable_values_shape{1, 50, 5, 5};
    const PartialShape filters_shape{3, 3, 5, 5};

    auto param0 = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto param1 = make_shared<op::Parameter>(element::f32, deformable_values_shape);
    auto param2 = make_shared<op::Parameter>(element::f32, filters_shape);

    try
    {
        make_shared<op::v1::DeformableConvolution>(param0,
                                                   param1,
                                                   param2,
                                                   Strides{},
                                                   CoordinateDiff{},
                                                   CoordinateDiff{},
                                                   Strides{},
                                                   op::PadType::EXPLICIT,
                                                   1,
                                                   7);

        FAIL() << "DeformableConvolution created with incorrect 'deformable group' value";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "deformable values input must be evenly divisible");
    }
}
