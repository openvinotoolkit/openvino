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

TEST(type_prop, group_conv)
{
    // Deduce type
    auto data = make_shared<op::Parameter>(element::f32, Shape{64, 4, 100, 150});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{128, 2, 10, 20});
    auto conv = make_shared<op::GroupConvolution>(data,
                                                  filters,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1},
                                                  2);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 91, 131}));
}

TEST(type_prop, group_conv_auto)
{
    // Deduce type
    auto data = make_shared<op::Parameter>(element::f32, Shape{64, 4, 100, 150});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{128, 2, 10, 20});
    auto conv = make_shared<op::GroupConvolution>(data,
                                                  filters,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1},
                                                  2,
                                                  op::PadType::AUTO);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 100, 150}));
    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{4, 9}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{5, 10}));
}

TEST(type_prop, group_conv_invalid_groups)
{
    // Deduce type
    try
    {
        auto conv = make_shared<op::GroupConvolution>(
            make_shared<op::Parameter>(element::f32, Shape{64, 20, 100, 150}),
            make_shared<op::Parameter>(element::f32, Shape{30, 10, 10, 20}),
            Strides{1, 1},
            Strides{1, 1},
            CoordinateDiff{0, 0},
            CoordinateDiff{0, 0},
            Strides{1, 1},
            3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid group conv";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data channels not a multiple of group size"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
    try
    {
        auto conv = make_shared<op::GroupConvolution>(
            make_shared<op::Parameter>(element::f32, Shape{64, 30, 100, 150}),
            make_shared<op::Parameter>(element::f32, Shape{20, 10, 10, 20}),
            Strides{1, 1},
            Strides{1, 1},
            CoordinateDiff{0, 0},
            CoordinateDiff{0, 0},
            Strides{1, 1},
            3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid group conv";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("# Filters not a multiple of group size"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
    try
    {
        auto conv = make_shared<op::GroupConvolution>(
            make_shared<op::Parameter>(element::f32, Shape{64, 30, 100, 150}),
            make_shared<op::Parameter>(element::f32, Shape{30, 20, 10, 20}),
            Strides{1, 1},
            Strides{1, 1},
            CoordinateDiff{0, 0},
            CoordinateDiff{0, 0},
            Strides{1, 1},
            3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid group conv";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incorrect number of channels per filter"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
