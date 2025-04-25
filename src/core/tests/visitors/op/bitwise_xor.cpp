// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_xor.hpp"

#include "binary_ops.hpp"

using Type = ::testing::Types<BinaryOperatorType<ov::op::v13::BitwiseXor, ov::element::i32>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_with_auto_broadcast, BinaryOperatorVisitor, Type, BinaryOperatorTypeName);
