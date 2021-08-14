// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/minimum.hpp"

#include "binary_ops.hpp"
#include "ngraph/type/element_type.hpp"

using Type = ::testing::Types<BinaryOperatorType<ngraph::op::v1::Minimum, ngraph::element::f32>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_with_auto_broadcast, BinaryOperatorVisitor, Type, BinaryOperatorTypeName);
