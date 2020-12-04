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

TEST(type_prop, shuffle_channels_axis_validation)
{
    try
    {
        const auto data = make_shared<op::Parameter>(element::f64, Shape{1, 2, 3, 4});
        const auto shuffle_channels = make_shared<op::ShuffleChannels>(data, -5, 5);
        FAIL() << "ShuffleChannels validation did not work. Op node was created with incorrect "
                  "params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The 'axis' parameter for ShuffleChannels has to point to one of the "
                             "input tensor's shape dimensions");
    }
}

TEST(type_prop, shuffle_channels_negative_axis_calculation)
{
    const auto data = make_shared<op::Parameter>(element::f64, Shape{1, 2, 3, 4});

    const auto shuffle_channels = make_shared<op::ShuffleChannels>(data, -3, 2);

    EXPECT_EQ(shuffle_channels->get_zero_based_axis(), 1);
}

TEST(type_prop, shuffle_channels_invalid_input_shape)
{
    try
    {
        const auto data = make_shared<op::Parameter>(element::f64, Shape{});
        const auto shuffle_channels = make_shared<op::ShuffleChannels>(data, 0, 1);
        FAIL() << "ShuffleChannels validation did not work. Op node was created with incorrect "
                  "params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The input tensor's shape is expected to be at least 1D.");
    }
}

TEST(type_prop, shuffle_channels_invalid_groups_value)
{
    try
    {
        const auto data = make_shared<op::Parameter>(element::f64, Shape{1, 2, 3, 15});
        const auto shuffle_channels = make_shared<op::ShuffleChannels>(data, -1, 2);
        FAIL() << "ShuffleChannels validation did not work. Op node was created with incorrect "
                  "params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "The channel dimension size has to be a multiple of the groups parameter value.");
    }
}
