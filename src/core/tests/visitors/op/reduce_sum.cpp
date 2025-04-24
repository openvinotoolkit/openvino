// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_sum.hpp"

#include "reduce_ops.hpp"

using Type = ::testing::Types<ReduceOperatorType<ov::op::v1::ReduceSum, ov::element::f32>>;
INSTANTIATE_TYPED_TEST_SUITE_P(attributes_reduce_op, ReduceOperatorVisitor, Type, ReduceOperatorTypeName);
