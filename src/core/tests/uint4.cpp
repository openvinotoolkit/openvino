// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(uint4, convert_u4_to_string) {
    vector<uint8_t> values{171, 16};
    auto constant = make_shared<op::Constant>(element::u4, Shape{3}, &values[0]);

    vector<string> ref{"10", "11", "1"};
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(constant->convert_value_to_string(i), ref[i]);
    }
}

TEST(uint4, tensor_or_constant_size) {
    vector<uint8_t> values{171, 16};
    auto constant = make_shared<op::Constant>(element::u4, Shape{3}, &values[0]);
    EXPECT_EQ(2, constant->get_byte_size());

    ngraph::HostTensor host_tensor(ngraph::element::u4, Shape{3});
    EXPECT_EQ(constant->get_byte_size(), host_tensor.get_size_in_bytes());

    ov::Tensor runtime_tensor(ov::element::u4, ov::Shape{3});
    EXPECT_EQ(constant->get_byte_size(), runtime_tensor.get_byte_size());
}

TEST(u1, tensor_or_constant_size) {
    vector<uint8_t> values{171, 16};
    auto constant = make_shared<op::Constant>(element::u1, Shape{3}, &values[0]);
    EXPECT_EQ(1, constant->get_byte_size());

    ngraph::HostTensor host_tensor(ngraph::element::u1, Shape{3});
    EXPECT_EQ(constant->get_byte_size(), host_tensor.get_size_in_bytes());

    ov::Tensor runtime_tensor(ov::element::u1, ov::Shape{3});
    EXPECT_EQ(constant->get_byte_size(), runtime_tensor.get_byte_size());
}
