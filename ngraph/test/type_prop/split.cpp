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

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(type_prop, split)
{
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});

    try
    {
        const std::vector<size_t> splits = {1, 6}; // should sum up to 6
        const auto axis = op::Constant::create(element::i64, Shape{}, {1});
        const auto split = make_shared<op::Split>(data, axis, splits);
        FAIL() << "Split node was created with incorrect data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("has to be equal to the sum of splits passed to the op: 7"));
    }

    try
    {
        const std::vector<size_t> splits = {4, 2};
        const auto axis = op::Constant::create(element::i64, Shape{}, {-5});
        const auto split = make_shared<op::Split>(data, axis, splits); // invalid axis
        FAIL() << "Split node was created with incorrect data.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Parameter axis -5 out of the tensor rank"));
    }

    const auto axis = op::Constant::create(element::i64, Shape{}, {1});
    const auto split = make_shared<op::Split>(data, axis, 2);
    EXPECT_EQ(split->outputs().size(), 2);
    EXPECT_EQ(split->get_output_shape(0), (Shape{2, 3}));
    EXPECT_EQ(split->get_output_shape(1), (Shape{2, 3}));
    EXPECT_EQ(split->get_output_element_type(0), element::i32);
    EXPECT_EQ(split->get_output_element_type(1), element::i32);
}

TEST(type_prop, split_axis_must_be_scalar)
{
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});
    const std::vector<size_t> splits = {1, 6};
    const auto axis = op::Constant::create(element::i64, Shape{2}, {0, 1});

    try
    {
        const auto split = make_shared<op::Split>(data, axis, splits);
        FAIL() << "Incorrect axis of Split not detected.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("The 'axis' input node must be scalar"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason.";
    }
}

TEST(type_prop, split_axis_must_be_constant)
{
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});
    const std::vector<size_t> splits = {1, 6};
    const auto axis = make_shared<op::Parameter>(element::i32, Shape{});

    try
    {
        const auto split = make_shared<op::Split>(data, axis, splits);
        FAIL() << "Not constant axis of Split not detected.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("The 'axis' input node must be constant"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason.";
    }
}
