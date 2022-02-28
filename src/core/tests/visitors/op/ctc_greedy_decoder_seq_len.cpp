// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, ctc_greedy_decoder_seq_len_op) {
    NodeBuilder::get_ops().register_factory<opset6::CTCGreedyDecoderSeqLen>();
    bool merge_repeated = false;
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 3});
    auto seq_len = make_shared<op::Parameter>(element::i32, Shape{1});
    auto blank_index = op::Constant::create<int32_t>(element::i32, Shape{}, {2});
    auto decoder = make_shared<op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, blank_index, merge_repeated);

    NodeBuilder builder(decoder);
    auto g_decoder = ov::as_type_ptr<opset6::CTCGreedyDecoderSeqLen>(builder.create());

    EXPECT_EQ(g_decoder->get_merge_repeated(), decoder->get_merge_repeated());
}
