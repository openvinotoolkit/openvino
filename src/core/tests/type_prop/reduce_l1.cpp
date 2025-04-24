// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_l1.hpp"

#include "reduce_ops.hpp"

using Type = ::testing::Types<ov::op::v4::ReduceL1>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_l1, ReduceTest, Type);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_l1_et, ReduceArithmeticTest, Type);
