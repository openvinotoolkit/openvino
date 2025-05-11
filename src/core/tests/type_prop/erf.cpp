// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/erf.hpp"

#include "unary_ops.hpp"

using Type = ::testing::Types<ov::op::v0::Erf>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_erf, UnaryOperator, Type);
