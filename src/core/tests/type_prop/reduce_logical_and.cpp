// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_logical_and.hpp"

#include "reduce_ops.hpp"

using Type = ::testing::Types<ov::op::v1::ReduceLogicalAnd>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_logical_and, ReduceTest, Type);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_logical_and_et, ReduceLogicalTest, Type);
