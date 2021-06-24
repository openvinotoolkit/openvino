// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unary_ops.hpp"

using Types = ::testing::Types<UnaryOperatorType<ngraph::op::v0::Floor, element::f32>,
                               UnaryOperatorType<ngraph::op::v0::Floor, element::f16>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_without_attribute,
                               UnaryOperatorVisitor,
                               Types,
                               UnaryOperatorTypeName);
