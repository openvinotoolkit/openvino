// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/power.hpp"

#include "arithmetic_ops.hpp"

using Type = ::testing::Types<ov::op::v1::Power>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_power, ArithmeticOperator, Type);
