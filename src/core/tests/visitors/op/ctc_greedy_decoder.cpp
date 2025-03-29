// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ctc_greedy_decoder.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, ctc_greedy_decoder_op) {
    NodeBuilder::opset().insert<ov::op::v0::CTCGreedyDecoder>();
    bool m_ctc_merge_repeated = false;
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 1, 3});
    auto masks = make_shared<ov::op::v0::Parameter>(element::i32, Shape{3, 1});
    auto decoder = make_shared<op::v0::CTCGreedyDecoder>(data, masks, m_ctc_merge_repeated);

    NodeBuilder builder(decoder, {data, masks});
    auto g_decoder = ov::as_type_ptr<ov::op::v0::CTCGreedyDecoder>(builder.create());

    EXPECT_EQ(g_decoder->get_ctc_merge_repeated(), decoder->get_ctc_merge_repeated());
}
