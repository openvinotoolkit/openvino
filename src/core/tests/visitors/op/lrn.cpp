// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lrn.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, lrn_op) {
    NodeBuilder::opset().insert<ov::op::v0::LRN>();
    const auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto axes = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});

    const double alpha = 1.1;
    const double beta = 2.2;
    const double bias = 3.3;
    const size_t size = 4;

    const auto lrn = make_shared<ov::op::v0::LRN>(arg, axes, alpha, beta, bias, size);
    NodeBuilder builder(lrn, {arg, axes});
    auto g_lrn = ov::as_type_ptr<ov::op::v0::LRN>(builder.create());

    EXPECT_EQ(g_lrn->get_alpha(), lrn->get_alpha());
    EXPECT_EQ(g_lrn->get_beta(), lrn->get_beta());
    EXPECT_EQ(g_lrn->get_bias(), lrn->get_bias());
    EXPECT_EQ(g_lrn->get_nsize(), lrn->get_nsize());
}
