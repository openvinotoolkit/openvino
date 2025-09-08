// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/pad.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

template <class T>
class PadAttrVisitorTest : public testing::Test {};

TYPED_TEST_SUITE_P(PadAttrVisitorTest);

TYPED_TEST_P(PadAttrVisitorTest, pad_basic) {
    NodeBuilder::opset().insert<TypeParam>();
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});

    auto pad_mode = op::PadMode::EDGE;

    auto pad = make_shared<TypeParam>(arg, pads_begin, pads_end, pad_mode);
    NodeBuilder builder(pad, {arg, pads_begin, pads_end});
    auto g_pad = ov::as_type_ptr<TypeParam>(builder.create());

    EXPECT_EQ(g_pad->get_pad_mode(), pad->get_pad_mode());
    EXPECT_EQ(g_pad->get_pads_begin(), pad->get_pads_begin());
    EXPECT_EQ(g_pad->get_pads_end(), pad->get_pads_end());
}

TYPED_TEST_P(PadAttrVisitorTest, pad_const_mode) {
    NodeBuilder::opset().insert<TypeParam>();
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});
    auto pad_value = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});

    auto pad_mode = op::PadMode::CONSTANT;

    auto pad = make_shared<TypeParam>(arg, pads_begin, pads_end, pad_value, pad_mode);
    NodeBuilder builder(pad, {arg, pads_begin, pads_end, pad_value});
    auto g_pad = ov::as_type_ptr<TypeParam>(builder.create());

    EXPECT_EQ(g_pad->get_pad_mode(), pad->get_pad_mode());
    EXPECT_EQ(g_pad->get_pads_begin(), pad->get_pads_begin());
    EXPECT_EQ(g_pad->get_pads_end(), pad->get_pads_end());
}

REGISTER_TYPED_TEST_SUITE_P(PadAttrVisitorTest, pad_basic, pad_const_mode);

using PadOpTypes = testing::Types<op::v1::Pad, op::v12::Pad>;
INSTANTIATE_TYPED_TEST_SUITE_P(attributes, PadAttrVisitorTest, PadOpTypes);
