// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

void static test_normalize_l2_attributes(float eps, op::EpsMode eps_mode) {
    NodeBuilder::get_ops().register_factory<opset1::NormalizeL2>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4});
    const auto axes = make_shared<op::Constant>(element::i32, Shape{}, vector<int32_t>{1});

    auto normalize_l2 = make_shared<opset1::NormalizeL2>(data, axes, eps, eps_mode);
    NodeBuilder builder(normalize_l2);
    auto g_normalize_l2 = ov::as_type_ptr<opset1::NormalizeL2>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_normalize_l2->get_eps(), normalize_l2->get_eps());
    EXPECT_EQ(g_normalize_l2->get_eps_mode(), normalize_l2->get_eps_mode());
}

TEST(attributes, normalize_l2_op_mode_add) {
    test_normalize_l2_attributes(1e-6f, op::EpsMode::ADD);
}

TEST(attributes, normalize_l2_op_mode_max) {
    test_normalize_l2_attributes(1e-3f, op::EpsMode::MAX);
}
