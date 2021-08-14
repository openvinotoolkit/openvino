// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/gelu.hpp"

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, gelu_op) {
    NodeBuilder::get_ops().register_factory<op::v7::Gelu>();
    const auto data_input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto approximation_mode = op::GeluApproximationMode::ERF;
    const auto gelu = make_shared<op::v7::Gelu>(data_input, approximation_mode);
    NodeBuilder builder(gelu);
    auto g_gelu = as_type_ptr<op::v7::Gelu>(builder.create());

    EXPECT_EQ(g_gelu->get_approximation_mode(), gelu->get_approximation_mode());
}
