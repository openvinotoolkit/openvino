// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/normalize_l2.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

void static test_normalize_l2_attributes(float eps, op::EpsMode eps_mode) {
    NodeBuilder::opset().insert<ov::op::v0::NormalizeL2>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3, 4});
    const auto axes = make_shared<ov::op::v0::Constant>(element::i32, Shape{}, vector<int32_t>{1});

    auto normalize_l2 = make_shared<ov::op::v0::NormalizeL2>(data, axes, eps, eps_mode);
    NodeBuilder builder(normalize_l2, {data, axes});
    auto g_normalize_l2 = ov::as_type_ptr<ov::op::v0::NormalizeL2>(builder.create());

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
