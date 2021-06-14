// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "unary_ops.hpp"

using Types = ::testing::Types<op::Acos, op::Asin, op::Abs, op::Sqrt, op::Sin, op::Exp, op::Floor>;

INSTANTIATE_TYPED_TEST_CASE_P(type_prop, UnaryOperator, Types);
