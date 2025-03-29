// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/logical_and.hpp"

#include "common_test_utils/type_prop.hpp"
#include "logical_ops.hpp"

using Type = ::testing::Types<LogicalOperatorType<ov::op::v1::LogicalAnd, ov::element::boolean>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Type_prop_test, LogicalOperatorTypeProp, Type, LogicalOperatorTypeName);
