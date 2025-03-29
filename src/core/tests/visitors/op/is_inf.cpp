// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/is_inf.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, is_inf_defaults) {
    NodeBuilder::opset().insert<ov::op::v10::IsInf>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 10, 10});

    const auto is_inf = make_shared<ov::op::v10::IsInf>(data);
    NodeBuilder builder(is_inf);

    auto g_is_inf = ov::as_type_ptr<ov::op::v10::IsInf>(builder.create());

    const auto is_inf_attrs = is_inf->get_attributes();
    const auto g_is_inf_attrs = g_is_inf->get_attributes();

    EXPECT_EQ(g_is_inf_attrs.detect_positive, is_inf_attrs.detect_positive);
    EXPECT_EQ(g_is_inf_attrs.detect_negative, is_inf_attrs.detect_negative);
}

TEST(attributes, is_inf_positive_only) {
    NodeBuilder::opset().insert<ov::op::v10::IsInf>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 10, 10});

    ov::op::v10::IsInf::Attributes attributes{};
    attributes.detect_negative = false;
    const auto is_inf = make_shared<ov::op::v10::IsInf>(data, attributes);
    NodeBuilder builder(is_inf);

    auto g_is_inf = ov::as_type_ptr<ov::op::v10::IsInf>(builder.create());

    const auto is_inf_attrs = is_inf->get_attributes();
    const auto g_is_inf_attrs = g_is_inf->get_attributes();

    EXPECT_EQ(g_is_inf_attrs.detect_positive, is_inf_attrs.detect_positive);
    EXPECT_EQ(g_is_inf_attrs.detect_negative, is_inf_attrs.detect_negative);
}

TEST(attributes, is_inf_negative_only) {
    NodeBuilder::opset().insert<ov::op::v10::IsInf>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 10, 10});

    ov::op::v10::IsInf::Attributes attributes{};
    attributes.detect_positive = false;
    const auto is_inf = make_shared<ov::op::v10::IsInf>(data, attributes);
    NodeBuilder builder(is_inf);

    auto g_is_inf = ov::as_type_ptr<ov::op::v10::IsInf>(builder.create());

    const auto is_inf_attrs = is_inf->get_attributes();
    const auto g_is_inf_attrs = g_is_inf->get_attributes();

    EXPECT_EQ(g_is_inf_attrs.detect_positive, is_inf_attrs.detect_positive);
    EXPECT_EQ(g_is_inf_attrs.detect_negative, is_inf_attrs.detect_negative);
}

TEST(attributes, is_inf_detect_none) {
    NodeBuilder::opset().insert<ov::op::v10::IsInf>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 10, 10});

    ov::op::v10::IsInf::Attributes attributes{};
    attributes.detect_negative = false;
    attributes.detect_positive = false;
    const auto is_inf = make_shared<ov::op::v10::IsInf>(data, attributes);
    NodeBuilder builder(is_inf);

    auto g_is_inf = ov::as_type_ptr<ov::op::v10::IsInf>(builder.create());

    const auto is_inf_attrs = is_inf->get_attributes();
    const auto g_is_inf_attrs = g_is_inf->get_attributes();

    EXPECT_EQ(g_is_inf_attrs.detect_positive, is_inf_attrs.detect_positive);
    EXPECT_EQ(g_is_inf_attrs.detect_negative, is_inf_attrs.detect_negative);
}
