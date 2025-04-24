// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/elu.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, elu_op) {
    NodeBuilder::opset().insert<ov::op::v0::Elu>();
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});

    double alpha = 0.1;

    const auto elu = make_shared<ov::op::v0::Elu>(data, alpha);
    NodeBuilder builder(elu, {data});
    auto g_elu = ov::as_type_ptr<ov::op::v0::Elu>(builder.create());

    EXPECT_EQ(g_elu->get_alpha(), elu->get_alpha());
}
