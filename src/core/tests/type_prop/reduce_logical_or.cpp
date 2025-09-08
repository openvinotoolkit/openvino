// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_logical_or.hpp"

#include "reduce_ops.hpp"

using Type = ::testing::Types<ov::op::v1::ReduceLogicalOr>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_logical_or, ReduceTest, Type);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_logical_or_et, ReduceLogicalTest, Type);
