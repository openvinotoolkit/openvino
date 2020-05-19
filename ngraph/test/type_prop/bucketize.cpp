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

TEST(type_prop, bucketize)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 2});
    auto buckets = make_shared<op::Parameter>(element::f32, Shape{4});
    auto bucketize = make_shared<op::v3::Bucketize>(data, buckets);
    EXPECT_EQ(bucketize->get_element_type(), element::i64);
    EXPECT_TRUE(bucketize->get_output_partial_shape(0).same_scheme(PartialShape{2, 3, 2}));
}

TEST(type_prop, bucketize_output_type)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto buckets = make_shared<op::Parameter>(element::f32, Shape{5});
    auto bucketize = make_shared<op::v3::Bucketize>(data, buckets, element::i32);

    ASSERT_EQ(bucketize->get_output_element_type(0), element::i32);
    EXPECT_TRUE(bucketize->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, 3, 4}));
}

TEST(type_prop, bucketize_output_type_right_bound)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto buckets = make_shared<op::Parameter>(element::f32, Shape{5});
    auto bucketize = make_shared<op::v3::Bucketize>(data, buckets, element::i32, false);

    ASSERT_EQ(bucketize->get_output_element_type(0), element::i32);
    EXPECT_TRUE(bucketize->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, 3, 4}));
}

TEST(type_prop, bucketize_dynamic_input)
{
    auto data = make_shared<op::Parameter>(element::f64, PartialShape{4, Dimension::dynamic()});
    auto buckets = make_shared<op::Parameter>(element::f32, Shape{5});
    auto bucketize = make_shared<op::v3::Bucketize>(data, buckets);

    EXPECT_EQ(bucketize->get_element_type(), element::i64);
    EXPECT_TRUE(
        bucketize->get_output_partial_shape(0).same_scheme(PartialShape{4, Dimension::dynamic()}));
}

TEST(type_prop, bucketize_dynamic_buckets)
{
    auto data = make_shared<op::Parameter>(element::f64, PartialShape{4, Dimension::dynamic()});
    auto buckets = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    auto bucketize = make_shared<op::v3::Bucketize>(data, buckets);

    EXPECT_EQ(bucketize->get_element_type(), element::i64);
    EXPECT_TRUE(
        bucketize->get_output_partial_shape(0).same_scheme(PartialShape{4, Dimension::dynamic()}));
}

TEST(type_prop, bucketize_fail_output_type)
{
    auto data = make_shared<op::Parameter>(element::f64, PartialShape{4, Dimension::dynamic()});
    auto buckets = make_shared<op::Parameter>(element::f32, Shape{5});
    try
    {
        auto bucketize = make_shared<op::v3::Bucketize>(data, buckets, element::f64);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid output type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Output type must be i32 or i64"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, bucketize_fail_buckets_dim)
{
    auto data = make_shared<op::Parameter>(element::f64, PartialShape{4, Dimension::dynamic()});
    auto buckets = make_shared<op::Parameter>(element::f32, Shape{5, 5});
    try
    {
        auto bucketize = make_shared<op::v3::Bucketize>(data, buckets);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid output type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("buckets input must be a 1D tensor"));
    }
    catch (...)
    {
        FAIL() << "buckets dimension check failed for unexpected reason";
    }
}
