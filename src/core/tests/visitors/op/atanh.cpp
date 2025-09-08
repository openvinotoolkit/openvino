// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/atanh.hpp"

#include "unary_ops.hpp"

using Types = ::testing::Types<UnaryOperatorType<ov::op::v3::Atanh, ov::element::f32>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_without_attribute, UnaryOperatorVisitor, Types, UnaryOperatorTypeName);
