// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ctc_greedy_decoder_seq_len.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, ctc_greedy_decoder_seq_len_op) {
    NodeBuilder::opset().insert<ov::op::v6::CTCGreedyDecoderSeqLen>();
    bool merge_repeated = false;
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 3});
    auto seq_len = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1});
    auto blank_index = ov::op::v0::Constant::create<int32_t>(element::i32, Shape{}, {2});
    auto decoder = make_shared<op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, blank_index, merge_repeated);

    NodeBuilder builder(decoder, {data, seq_len, blank_index});
    auto g_decoder = ov::as_type_ptr<ov::op::v6::CTCGreedyDecoderSeqLen>(builder.create());

    EXPECT_EQ(g_decoder->get_merge_repeated(), decoder->get_merge_repeated());
}
