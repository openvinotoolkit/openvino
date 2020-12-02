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

TEST(type_prop, scalar_constant_deduce_float32)
{
    auto c = op::Constant::create(element::Type_t::f32, Shape{}, {208});
    ASSERT_EQ(c->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(c->get_shape(), (Shape{}));
}

TEST(type_prop, scalar_constant_deduce_bool)
{
    auto c = op::Constant::create(element::Type_t::boolean, Shape{}, {1});
    ASSERT_EQ(c->get_element_type(), element::Type_t::boolean);
    ASSERT_EQ(c->get_shape(), (Shape{}));
}

TEST(type_prop, tensor_constant_deduce_float32)
{
    auto c = op::Constant::create(element::Type_t::f32, Shape{2, 2}, {208, 208, 208, 208});
    ASSERT_EQ(c->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 2}));
}

TEST(type_prop, tensor_constant_deduce_bool)
{
    auto c = op::Constant::create(element::Type_t::boolean, Shape{2, 2}, {1, 1, 1, 1});
    ASSERT_EQ(c->get_element_type(), element::Type_t::boolean);
    ASSERT_EQ(c->get_shape(), (Shape{2, 2}));
}

TEST(type_prop, tensor_constant_bad_count)
{
    try
    {
        auto c = op::Constant::create(element::Type_t::boolean, Shape{2, 2}, {1, 1, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect number of literals not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Did not get the expected number of literals for a "
                                         "constant of shape Shape{2, 2} (got 3, expected 1 or 4)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, constant_zero_elements_one_string)
{
    auto c = make_shared<op::Constant>(
        element::Type_t::i64, Shape{2, 0, 2, 2}, std::vector<std::string>{"42"});
    ASSERT_EQ(c->get_element_type(), element::Type_t::i64);
    ASSERT_EQ(c->get_shape(), (Shape{2, 0, 2, 2}));
}
