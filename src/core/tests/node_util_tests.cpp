// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/util/node_util.hpp"

TEST(NodeUtilTests, set_name_parameter) {
    std::string name = "test_name";
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape{1});
    ov::op::util::set_name(*param, name);
    EXPECT_TRUE(0 == name.compare(param->get_friendly_name()));
    EXPECT_TRUE(0 == name.compare(*param->output(0).get_names().begin()));
}

TEST(NodeUtilTests, set_name_result) {
    std::string name = "test_name";
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape{1});
    auto result = std::make_shared<ov::op::v0::Result>(param);
    ov::op::util::set_name(*result, name);
    EXPECT_TRUE(0 == name.compare(result->get_friendly_name()));
    EXPECT_TRUE(0 == name.compare(*result->output(0).get_names().begin()));
}

TEST(NodeUtilTests, set_name_split) {
    std::string name = "test_name";
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape{2, 2});
    auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape{});
    auto split = std::make_shared<ov::op::v1::Split>(param1, param2, 2);
    ov::op::util::set_name(*split, name, 1);
    EXPECT_TRUE(0 == name.compare(split->get_friendly_name()));
    EXPECT_EQ(0, split->output(0).get_names().size());
    EXPECT_TRUE(0 == name.compare(*split->output(1).get_names().begin()));
}
