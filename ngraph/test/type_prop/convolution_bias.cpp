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

TEST(type_prop, conv_bias_2d_deduce)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{128});
    auto conv = make_shared<op::ConvolutionBias>(param0, param1, param2);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 91, 131}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_bias_add_2d_deduce)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{128});
    auto param3 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 91, 131});
    auto conv = make_shared<op::ConvolutionBiasAdd>(param0,
                                                    param1,
                                                    param2,
                                                    param3,
                                                    Strides{1, 1},
                                                    Strides{1, 1},
                                                    CoordinateDiff{0, 0},
                                                    CoordinateDiff{0, 0},
                                                    Strides{1, 1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 91, 131}));
}
