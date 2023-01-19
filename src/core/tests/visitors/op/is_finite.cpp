// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset10.hpp"
#include "unary_ops.hpp"

using Types = ::testing::Types<UnaryOperatorType<ov::op::v10::IsFinite, ngraph::element::f32>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_without_attribute, UnaryOperatorVisitor, Types, UnaryOperatorTypeName);
