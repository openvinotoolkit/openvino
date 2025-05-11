// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/multiply.hpp"

#include "arithmetic_ops.hpp"

using Type = ::testing::Types<ov::op::v1::Multiply>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_multiply, ArithmeticOperator, Type);
