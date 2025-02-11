// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "binary_ops.hpp"
#include "openvino/op/bitwise_right_shift.hpp"

using Type = ::testing::Types<BinaryOperatorType<ov::op::v15::BitwiseRightShift, ov::element::i32>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_with_auto_broadcast, BinaryOperatorVisitor, Type, BinaryOperatorTypeName);
