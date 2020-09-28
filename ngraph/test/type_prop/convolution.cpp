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

#include <runtime/op/convolution.hpp>
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
    auto conv = make_shared<op::v1::Convolution>(
        param0, param1, Strides{1}, CoordinateDiff{0}, CoordinateDiff{0}, Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 91}));

    EXPECT_EQ(conv->get_strides(), Strides{1});
    EXPECT_EQ(conv->get_dilations(), Strides{1});

    EXPECT_EQ(conv->get_pads_begin(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_pads_end(), CoordinateDiff{0});
}

TEST(type_prop, conv_1d_back_data_batch_deduce)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});        // filters
    auto param_filters = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10}); // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 91}); // output delta
    auto conv = make_shared<op::v0::ConvolutionBackpropData>(data_batch_shape,
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
// TODO: Requires complete rewriting without v0 ops usage