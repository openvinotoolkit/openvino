// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, squared_difference)
{
    const auto x1 = make_shared<op::Parameter>(element::f64, Shape{2, 2});
    const auto x2 = make_shared<op::Parameter>(element::f64, Shape{3, 2});
    const auto x3 = make_shared<op::Parameter>(element::f64, Shape{1, 2});

    try
    {
        const auto squared_diff = make_shared<op::SquaredDifference>(x1, x2);
        FAIL() << "SquaredDifference node was created with incorrect data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument shapes are inconsistent"));
    }

    const auto squared_diff = make_shared<op::SquaredDifference>(x1, x3);
    EXPECT_EQ(squared_diff->get_element_type(), element::f64);
    EXPECT_EQ(squared_diff->get_shape(), (Shape{2, 2}));
    EXPECT_EQ(squared_diff->get_autob(), op::AutoBroadcastType::NUMPY);

    const auto squared_diff_no_args = make_shared<op::SquaredDifference>();
    EXPECT_EQ(squared_diff_no_args->get_autob(), op::AutoBroadcastType::NUMPY);
}
