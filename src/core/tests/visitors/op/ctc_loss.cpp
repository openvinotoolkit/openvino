// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, ctc_loss) {
    NodeBuilder::get_ops().register_factory<opset4::CTCLoss>();

    auto logits = make_shared<op::Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto labels = make_shared<op::Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto blank_index = make_shared<op::Parameter>(element::i32, Shape{});

    auto ctc_loss = make_shared<opset4::CTCLoss>(logits, logit_length, labels, label_length, blank_index);
    NodeBuilder builder(ctc_loss);
    auto g_ctc_loss = as_type_ptr<opset4::CTCLoss>(builder.create());

    // attribute count
    const auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    // CTC Loss attributes
    EXPECT_EQ(g_ctc_loss->get_preprocess_collapse_repeated(), ctc_loss->get_preprocess_collapse_repeated());
    EXPECT_EQ(g_ctc_loss->get_ctc_merge_repeated(), ctc_loss->get_ctc_merge_repeated());
    EXPECT_EQ(g_ctc_loss->get_unique(), ctc_loss->get_unique());
}
