// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arithmetic_ops.hpp"

using Type = ::testing::Types<ngraph::op::v1::Mod>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_mod, ArithmeticOperator, Type);
