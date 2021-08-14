// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "binary_ops.hpp"
#include "ngraph/op/and.hpp"

using Type = ::testing::Types<BinaryOperatorType<ngraph::op::v1::LogicalAnd, ngraph::element::boolean>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_with_auto_broadcast, BinaryOperatorVisitor, Type, BinaryOperatorTypeName);
