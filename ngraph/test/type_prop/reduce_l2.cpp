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

TEST(type_prop, reduce_l2_v4_axis_out_of_range)
{
    auto arg = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 2, 3});
    auto axes = make_shared<op::Constant>(element::Type_t::i64, Shape{2}, vector<int64_t>{2, 3});
    try
    {
        auto reduce_sum = make_shared<op::v4::ReduceL2>(arg, axes);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect axes values exception not thrown";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reduction axis ("));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reduce_l2_v4_shape_if_keep_dims)
{
    auto arg = make_shared<op::Parameter>(element::Type_t::f32, Shape{3, 4, 5});
    auto axes = make_shared<op::Constant>(element::Type_t::i64, Shape{2}, vector<int64_t>{1, 2});
    auto keep_dims = true;
    auto reduce_prod = make_shared<op::v4::ReduceL2>(arg, axes, keep_dims);
    ASSERT_TRUE(reduce_prod->get_output_partial_shape(0).compatible(PartialShape{3, 1, 1}));
}

TEST(type_prop, reduce_l2_v4_shape_if_not_keep_dims)
{
    auto arg = make_shared<op::Parameter>(element::Type_t::f32, Shape{3, 4, 5});
    auto axes = make_shared<op::Constant>(element::Type_t::i64, Shape{2}, vector<int64_t>{1, 2});
    auto keep_dims = false;
    auto reduce_prod = make_shared<op::v4::ReduceL2>(arg, axes, keep_dims);
    ASSERT_TRUE(reduce_prod->get_output_partial_shape(0).compatible(PartialShape{3}));
}
