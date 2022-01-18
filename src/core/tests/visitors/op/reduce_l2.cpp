// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_ops.hpp"

using Type = ::testing::Types<ReduceOperatorType<ngraph::op::v4::ReduceL2, ngraph::element::f32>>;
INSTANTIATE_TYPED_TEST_SUITE_P(attributes_reduce_op, ReduceOperatorVisitor, Type, ReduceOperatorTypeName);
