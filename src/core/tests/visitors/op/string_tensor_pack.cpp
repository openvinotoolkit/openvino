// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/string_tensor_pack.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

TEST(attributes, string_tensor_pack_op) {
    ov::test::NodeBuilder::opset().insert<ov::op::v15::StringTensorPack>();
    const auto begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{3});
    const auto ends = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{3});
    const auto symbols = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{3});
    const auto string_tensor_pack = std::make_shared<ov::op::v15::StringTensorPack>(begins, ends, symbols);
    ov::test::NodeBuilder builder(string_tensor_pack, {begins, ends, symbols});
    ASSERT_NO_THROW(std::ignore = ov::as_type_ptr<ov::op::v15::StringTensorPack>(builder.create()));
    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
