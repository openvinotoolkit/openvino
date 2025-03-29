// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ctc_loss.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, ctc_loss) {
    NodeBuilder::opset().insert<ov::op::v4::CTCLoss>();

    auto logits = make_shared<ov::op::v0::Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<ov::op::v0::Parameter>(element::i32, Shape{10});
    auto labels = make_shared<ov::op::v0::Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<ov::op::v0::Parameter>(element::i32, Shape{10});
    auto blank_index = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});

    auto ctc_loss = make_shared<ov::op::v4::CTCLoss>(logits, logit_length, labels, label_length, blank_index);
    NodeBuilder builder(ctc_loss, {logits, logit_length, labels, label_length, blank_index});
    auto g_ctc_loss = as_type_ptr<ov::op::v4::CTCLoss>(builder.create());

    // attribute count
    const auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    // CTC Loss attributes
    EXPECT_EQ(g_ctc_loss->get_preprocess_collapse_repeated(), ctc_loss->get_preprocess_collapse_repeated());
    EXPECT_EQ(g_ctc_loss->get_ctc_merge_repeated(), ctc_loss->get_ctc_merge_repeated());
    EXPECT_EQ(g_ctc_loss->get_unique(), ctc_loss->get_unique());
}
