// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/framework_node.hpp"

#include <gtest/gtest.h>

#include <vector>

TEST(framework_node, attrs) {
    ov::op::util::FrameworkNodeAttrs attrs;

    attrs.set_opset_name("opset_name");
    ASSERT_EQ(attrs.get_opset_name(), "opset_name");

    attrs.set_type_name("type_name");
    ASSERT_EQ(attrs.get_type_name(), "type_name");

    attrs["attr1"] = "value1";
    ASSERT_EQ(attrs.at("attr1"), "value1");
    ASSERT_EQ(attrs.begin()->first, "attr1");
    ASSERT_EQ(attrs.begin()->first, "attr1");
    ASSERT_EQ(attrs.begin()->second, "value1");

    ov::op::util::FrameworkNodeAttrs a1, a2;
    a1.set_type_name("type_name");
    a2.set_type_name("type_name_");
    ASSERT_FALSE(a1 == a2);
    a2.set_type_name("type_name");
    ASSERT_TRUE(a1 == a2);
    a1.set_opset_name("opset_name");
    a2.set_opset_name("opset_name_");
    ASSERT_FALSE(a1 == a2);
    a2.set_opset_name("opset_name");
    ASSERT_TRUE(a1 == a2);
    a1["attr"] = "value";
    ASSERT_FALSE(a1 == a2);
    a2["attr"] = "value_";
    ASSERT_FALSE(a1 == a2);
    a2["attr"] = "value";
    ASSERT_TRUE(a1 == a2);
}
