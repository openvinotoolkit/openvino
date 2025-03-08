// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/identity.hpp"

#include "unary_ops.hpp"

using Type = ::testing::Types<UnaryOperatorType<ov::op::v16::Identity, ov::element::f32>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_without_attribute, UnaryOperatorVisitor, Type, UnaryOperatorTypeName);
