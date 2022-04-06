// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "openvino/runtime/tensor.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(int4, convert_i4_to_string) {
    vector<uint8_t> values{171, 16};
    auto constant = make_shared<op::Constant>(element::i4, Shape{3}, &values[0]);

    vector<string> ref{"-6", "-5", "1"};
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(constant->convert_value_to_string(i), ref[i]);
    }
}

TEST(int4, tensor_or_constant_size) {
    vector<uint8_t> values{171, 16};
    auto constant = make_shared<op::Constant>(element::i4, Shape{3}, &values[0]);
    EXPECT_EQ(2, constant->get_byte_size());

    ngraph::HostTensor host_tensor(ngraph::element::i4, Shape{3});
    EXPECT_EQ(constant->get_byte_size(), host_tensor.get_size_in_bytes());

    ov::Tensor runtime_tensor(ov::element::i4, ov::Shape{3});
    EXPECT_EQ(constant->get_byte_size(), runtime_tensor.get_byte_size());
}
