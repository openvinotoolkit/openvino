// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/floor_mod.hpp"

#include "arithmetic_ops.hpp"

using Type = ::testing::Types<ov::op::v1::FloorMod>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_floormod, ArithmeticOperator, Type);
