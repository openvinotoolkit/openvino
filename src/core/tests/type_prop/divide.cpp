// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/divide.hpp"

#include "arithmetic_ops.hpp"

using Type = ::testing::Types<ov::op::v1::Divide>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_divide, ArithmeticOperator, Type);
