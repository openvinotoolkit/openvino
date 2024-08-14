// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_left_shift.hpp"

#include "binary_ops.hpp"

using Type = ::testing::Types<BinaryOperatorType<ov::op::v15::BitwiseLeftShift, ov::element::i32>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_with_auto_broadcast, BinaryOperatorVisitor, Type, BinaryOperatorTypeName);
