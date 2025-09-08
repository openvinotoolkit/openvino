// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/reverse.hpp"
#include "utils.hpp"

TEST(shape_inference_utils_test, get_input_bounds_not_valid_port) {
    ov::op::v1::Reverse dummy_op;
    const size_t not_valid_port = 100;
    const ov::ITensorAccessor& ta = ov::make_tensor_accessor();

    const auto ret = ov::op::get_input_bounds<ov::PartialShape, int64_t>(&dummy_op, not_valid_port, ta);
    ASSERT_FALSE(ret);
}