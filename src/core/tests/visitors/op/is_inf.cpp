// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/opsets/opset10.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ov;
using ngraph::test::NodeBuilder;
using namespace ov::opset10;

TEST(attributes, is_inf_defaults) {
    NodeBuilder::get_ops().register_factory<IsInf>();
    const auto data = make_shared<Parameter>(element::f32, Shape{1, 3, 10, 10});

    const auto is_inf = make_shared<IsInf>(data);
    NodeBuilder builder(is_inf);

    auto g_is_inf = ov::as_type_ptr<IsInf>(builder.create());

    const auto is_inf_attrs = is_inf->get_attributes();
    const auto g_is_inf_attrs = g_is_inf->get_attributes();

    EXPECT_EQ(g_is_inf_attrs.detect_positive, is_inf_attrs.detect_positive);
    EXPECT_EQ(g_is_inf_attrs.detect_negative, is_inf_attrs.detect_negative);
}

TEST(attributes, is_inf_positive_only) {
    NodeBuilder::get_ops().register_factory<IsInf>();
    const auto data = make_shared<Parameter>(element::f32, Shape{1, 3, 10, 10});

    IsInf::Attributes attributes{};
    attributes.detect_negative = false;
    const auto is_inf = make_shared<IsInf>(data, attributes);
    NodeBuilder builder(is_inf);

    auto g_is_inf = ov::as_type_ptr<IsInf>(builder.create());

    const auto is_inf_attrs = is_inf->get_attributes();
    const auto g_is_inf_attrs = g_is_inf->get_attributes();

    EXPECT_EQ(g_is_inf_attrs.detect_positive, is_inf_attrs.detect_positive);
    EXPECT_EQ(g_is_inf_attrs.detect_negative, is_inf_attrs.detect_negative);
}

TEST(attributes, is_inf_negative_only) {
    NodeBuilder::get_ops().register_factory<IsInf>();
    const auto data = make_shared<Parameter>(element::f32, Shape{1, 3, 10, 10});

    IsInf::Attributes attributes{};
    attributes.detect_positive = false;
    const auto is_inf = make_shared<IsInf>(data, attributes);
    NodeBuilder builder(is_inf);

    auto g_is_inf = ov::as_type_ptr<IsInf>(builder.create());

    const auto is_inf_attrs = is_inf->get_attributes();
    const auto g_is_inf_attrs = g_is_inf->get_attributes();

    EXPECT_EQ(g_is_inf_attrs.detect_positive, is_inf_attrs.detect_positive);
    EXPECT_EQ(g_is_inf_attrs.detect_negative, is_inf_attrs.detect_negative);
}

TEST(attributes, is_inf_detect_none) {
    NodeBuilder::get_ops().register_factory<IsInf>();
    const auto data = make_shared<Parameter>(element::f32, Shape{1, 3, 10, 10});

    IsInf::Attributes attributes{};
    attributes.detect_negative = false;
    attributes.detect_positive = false;
    const auto is_inf = make_shared<IsInf>(data, attributes);
    NodeBuilder builder(is_inf);

    auto g_is_inf = ov::as_type_ptr<IsInf>(builder.create());

    const auto is_inf_attrs = is_inf->get_attributes();
    const auto g_is_inf_attrs = g_is_inf->get_attributes();

    EXPECT_EQ(g_is_inf_attrs.detect_positive, is_inf_attrs.detect_positive);
    EXPECT_EQ(g_is_inf_attrs.detect_negative, is_inf_attrs.detect_negative);
}
