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

TEST(type_prop, embedding_lookup_non_matrix_weights)
{
    auto tv0_2_4_param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});
    auto tv0_2_4_param_1 = make_shared<op::Parameter>(element::boolean, Shape{2, 4, 5});
    try
    {
        auto bc = make_shared<op::EmbeddingLookup>(tv0_2_4_param_0, tv0_2_4_param_1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect incorrect element types for arithmetic operator";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("weights are expected to be a matrix"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, embedding_lookup_static_shapes)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{8, 10, 12});
    auto weights = make_shared<op::Parameter>(element::f32, Shape{5, 10});
    auto embed = make_shared<op::EmbeddingLookup>(data, weights);
    ASSERT_EQ(embed->get_element_type(), element::f32);
    ASSERT_EQ(embed->get_shape(), (Shape{8, 10, 12, 10}));
}

TEST(type_prop, embedding_lookup_dynamic_shape_arg0)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto weights = make_shared<op::Parameter>(element::f32, Shape{5, 10});
    auto embed = make_shared<op::EmbeddingLookup>(data, weights);
    ASSERT_EQ(embed->get_element_type(), element::f32);
    ASSERT_TRUE(embed->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, embedding_lookup_dynamic_shape_arg1)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{8, 10, 12});
    auto weights = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto embed = make_shared<op::EmbeddingLookup>(data, weights);
    ASSERT_EQ(embed->get_element_type(), element::f32);
    PartialShape expected{8, 10, 12, Dimension::dynamic()};
    ASSERT_TRUE(embed->get_output_partial_shape(0).same_scheme(expected));
}

TEST(type_prop, embedding_lookup_shape_arg1_dynamic_embedding_length)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{8, 10, 12});
    auto weights = make_shared<op::Parameter>(element::f32, PartialShape{5, Dimension::dynamic()});
    auto embed = make_shared<op::EmbeddingLookup>(data, weights);
    ASSERT_EQ(embed->get_element_type(), element::f32);
    PartialShape expected{8, 10, 12, Dimension::dynamic()};
    ASSERT_TRUE(embed->get_output_partial_shape(0).same_scheme(expected));
}
