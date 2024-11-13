// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"

#include "unary_ops.hpp"

namespace v0 {
using Types = ::testing::Types<UnaryOperatorType<ov::op::v0::Squeeze, ov::element::f32>,
                               UnaryOperatorType<ov::op::v0::Squeeze, ov::element::f16>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_without_attribute, UnaryOperatorVisitor, Types, UnaryOperatorTypeName);
}  // namespace v0

namespace v15 {
using Types = ::testing::Types<UnaryOperatorTypeWithAttribute<ov::op::v15::Squeeze, ov::element::f32>,
                               UnaryOperatorTypeWithAttribute<ov::op::v15::Squeeze, ov::element::f16>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_single_attribute, UnaryOperatorVisitor, Types, UnaryOperatorTypeName);
}  // namespace v15
