// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/maximum.hpp"

#include "binary_ops.hpp"

using Type = ::testing::Types<BinaryOperatorType<ov::op::v1::Maximum, ov::element::f32>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_with_auto_broadcast, BinaryOperatorVisitor, Type, BinaryOperatorTypeName);
