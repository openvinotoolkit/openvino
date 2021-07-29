// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset7.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, gelu_op)
{
    NodeBuilder::get_ops().register_factory<opset7::Gelu>();
	const auto data_input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto approximation_mode = op::GeluApproximationMode::ERF;
    const auto gelu = make_shared<opset7::Gelu>(data_input, approximation_mode);
    NodeBuilder builder(gelu);
    auto g_gelu = as_type_ptr<opset7::Gelu>(builder.create());

    EXPECT_EQ(g_gelu->get_approximation_mode(), gelu->get_approximation_mode());
}
