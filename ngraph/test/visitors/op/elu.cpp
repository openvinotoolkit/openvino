// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/op/elu.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, elu_op) {
    NodeBuilder::get_ops().register_factory<op::v0::Elu>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4});

    double alpha = 0.1;

    const auto elu = make_shared<op::v0::Elu>(data, alpha);
    NodeBuilder builder(elu);
    auto g_elu = as_type_ptr<op::v0::Elu>(builder.create());

    EXPECT_EQ(g_elu->get_alpha(), elu->get_alpha());
}
