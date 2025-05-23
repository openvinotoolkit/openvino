// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/string_tensor_unpack.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

TEST(attributes, string_tensor_unpack_op) {
    ov::test::NodeBuilder::opset().insert<ov::op::v15::StringTensorUnpack>();
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::string, ov::Shape{4});
    const auto string_tensor_unpack = std::make_shared<ov::op::v15::StringTensorUnpack>(data);
    ov::test::NodeBuilder builder(string_tensor_unpack, {data});
    ASSERT_NO_THROW(std::ignore = ov::as_type_ptr<ov::op::v15::StringTensorUnpack>(builder.create()));
    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
