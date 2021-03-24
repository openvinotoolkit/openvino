// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
