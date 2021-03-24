// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, fused_clamp)
{
    const auto data = make_shared<op::Parameter>(element::f64, Shape{2, 2});

    try
    {
        const auto clamp = make_shared<op::Clamp>(data, 2.0, 1.0);
        EXPECT_FALSE(clamp.get())
            << "Clamp validation did not work. Op node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("The 'min' parameter needs to be less than 'max' for Clamp"));
    }

    const auto clamp = make_shared<op::Clamp>(data, 1.0, 2.0);
    EXPECT_EQ(clamp->get_element_type(), element::f64);
    EXPECT_EQ(clamp->get_shape(), (Shape{2, 2}));
}
