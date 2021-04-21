// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arithmetic_ops.hpp"

using Type = ::testing::Types<ngraph::op::FloorMod>;

INSTANTIATE_TYPED_TEST_CASE_P(type_prop_floormod, ArithmeticOperator, Type);
