// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_left_shift.hpp"

#include "bitwise_ops.hpp"

using OpType = ::testing::Types<ov::op::v15::BitwiseLeftShift>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_bitwise_left_shift, BitwiseOperator, OpType);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_bitwise_left_shift, BitwiseOperatorNotBoolean, OpType);
