// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/atan2.hpp"

#include "binary_ops.hpp"

using Type = ::testing::Types<BinaryOperatorType<ov::op::v17::Atan2, ov::element::f32>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_with_auto_broadcast, BinaryOperatorVisitor, Type, BinaryOperatorTypeName);
