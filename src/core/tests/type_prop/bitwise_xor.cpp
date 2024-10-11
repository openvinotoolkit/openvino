// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_xor.hpp"

#include "bitwise_ops.hpp"

using Type = ::testing::Types<ov::op::v13::BitwiseXor>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_bitwise_xor, BitwiseOperator, Type);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_bitwise_xor, BitwiseOperatorBoolean, Type);
