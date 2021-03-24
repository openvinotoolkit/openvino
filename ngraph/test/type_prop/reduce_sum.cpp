// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, reduce_sum_v1_axis_out_of_range)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{2, 3});
    try
    {
        auto reduce_sum = make_shared<op::v1::ReduceSum>(arg, axes);

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

TEST(type_prop, reduce_sum_v1_shape_if_keep_dims)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 2});
    auto keep_dims = true;
    auto reduce_prod = make_shared<op::v1::ReduceSum>(arg, axes, keep_dims);
    ASSERT_TRUE(reduce_prod->get_output_partial_shape(0).compatible(PartialShape{3, 1, 1}));
}

TEST(type_prop, reduce_sum_v1_shape_if_not_keep_dims)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 2});
    auto keep_dims = false;
    auto reduce_prod = make_shared<op::v1::ReduceSum>(arg, axes, keep_dims);
    ASSERT_TRUE(reduce_prod->get_output_partial_shape(0).compatible(PartialShape{3}));
}
