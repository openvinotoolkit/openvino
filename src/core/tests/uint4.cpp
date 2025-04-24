// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_tools.hpp"
#include "gtest/gtest.h"
#include "openvino/op/constant.hpp"

using namespace ov;
using namespace std;

TEST(uint4, convert_u4_to_string) {
    vector<uint8_t> values{171, 16};
    auto constant = make_shared<ov::op::v0::Constant>(element::u4, Shape{3}, &values[0]);

    vector<string> ref{"11", "10", "0"};
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(constant->convert_value_to_string(i), ref[i]);
    }
}

TEST(uint4, tensor_or_constant_size) {
    vector<uint8_t> values{171, 16};
    auto constant = make_shared<op::v0::Constant>(element::u4, Shape{3}, &values[0]);
    EXPECT_EQ(2, constant->get_byte_size());

    ov::Tensor runtime_tensor(ov::element::u4, ov::Shape{3});
    EXPECT_EQ(constant->get_byte_size(), runtime_tensor.get_byte_size());
}

TEST(u1, tensor_or_constant_size) {
    vector<uint8_t> values{171, 16};
    auto constant = make_shared<op::v0::Constant>(element::u1, Shape{3}, &values[0]);
    EXPECT_EQ(1, constant->get_byte_size());

    ov::Tensor runtime_tensor(ov::element::u1, ov::Shape{3});
    EXPECT_EQ(constant->get_byte_size(), runtime_tensor.get_byte_size());
}
