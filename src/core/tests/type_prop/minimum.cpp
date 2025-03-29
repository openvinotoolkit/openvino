// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/minimum.hpp"

#include "arithmetic_ops.hpp"

using Type = ::testing::Types<ov::op::v1::Minimum>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_minimum, ArithmeticOperator, Type);
