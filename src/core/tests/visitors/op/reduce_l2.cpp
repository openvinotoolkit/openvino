// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_l2.hpp"

#include "reduce_ops.hpp"

using Type = ::testing::Types<ReduceOperatorType<ov::op::v4::ReduceL2, ov::element::f32>>;
INSTANTIATE_TYPED_TEST_SUITE_P(attributes_reduce_op, ReduceOperatorVisitor, Type, ReduceOperatorTypeName);
