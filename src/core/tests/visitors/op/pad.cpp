// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/visitor.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

template <class T>
class PadAttrVisitorTest : public testing::Test {};

TYPED_TEST_SUITE_P(PadAttrVisitorTest);

TYPED_TEST_P(PadAttrVisitorTest, pad_basic) {
    NodeBuilder::get_ops().register_factory<TypeParam>();
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1});

    auto pad_mode = op::PadMode::EDGE;

    auto pad = make_shared<TypeParam>(arg, pads_begin, pads_end, pad_mode);
    NodeBuilder builder(pad, {arg, pads_begin, pads_end});
    auto g_pad = ov::as_type_ptr<TypeParam>(builder.create());

    EXPECT_EQ(g_pad->get_pad_mode(), pad->get_pad_mode());
    EXPECT_EQ(g_pad->get_pads_begin(), pad->get_pads_begin());
    EXPECT_EQ(g_pad->get_pads_end(), pad->get_pads_end());
}

TYPED_TEST_P(PadAttrVisitorTest, pad_const_mode) {
    NodeBuilder::get_ops().register_factory<TypeParam>();
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pad_value = make_shared<op::Parameter>(element::f32, Shape{});

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
