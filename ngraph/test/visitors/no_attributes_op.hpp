// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"

#include <memory>

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;


template <class T>
class NoAttributesUnaryOp : public testing::Test
{

};

TYPED_TEST_CASE_P(NoAttributesUnaryOp);

TYPED_TEST_P(NoAttributesUnaryOp, value_map_size)
{
    NodeBuilder::get_ops().register_factory<TypeParam>();
    const auto input = std::make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto op = std::make_shared<TypeParam>(input);

    NodeBuilder builder(op);
    const auto expected_attr_count = 0;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

REGISTER_TYPED_TEST_CASE_P(NoAttributesUnaryOp,
                           value_map_size);
