// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unary_ops.hpp"

using Type = ::testing::Types<ngraph::op::Ceiling>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_ceiling, UnaryOperator, Type);
