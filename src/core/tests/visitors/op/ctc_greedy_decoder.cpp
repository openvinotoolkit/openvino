// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, ctc_greedy_decoder_op) {
    NodeBuilder::get_ops().register_factory<opset1::CTCGreedyDecoder>();
    bool m_ctc_merge_repeated = false;
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 1, 3});
    auto masks = make_shared<op::Parameter>(element::i32, Shape{3, 1});
    auto decoder = make_shared<op::v0::CTCGreedyDecoder>(data, masks, m_ctc_merge_repeated);

    NodeBuilder builder(decoder);
    auto g_decoder = ov::as_type_ptr<opset1::CTCGreedyDecoder>(builder.create());

    EXPECT_EQ(g_decoder->get_ctc_merge_repeated(), decoder->get_ctc_merge_repeated());
}
