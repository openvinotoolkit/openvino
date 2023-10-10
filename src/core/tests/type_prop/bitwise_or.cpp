// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_or.hpp"

#include "bitwise_ops.hpp"

using Type = ::testing::Types<ov::op::v13::BitwiseOr>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_bitwise_or, BitwiseOperator, Type);
