// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "binary_ops.hpp"
#include "openvino/op/greater_eq.hpp"

using Type = ::testing::Types<BinaryOperatorType<ov::op::v1::GreaterEqual, ov::element::f32>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_with_auto_broadcast, BinaryOperatorVisitor, Type, BinaryOperatorTypeName);
