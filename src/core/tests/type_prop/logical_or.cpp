// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "logical_ops.hpp"
#include "util/type_prop.hpp"

using Type = ::testing::Types<LogicalOperatorType<ngraph::op::v1::LogicalOr, ngraph::element::boolean>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Type_prop_test, LogicalOperatorTypeProp, Type, LogicalOperatorTypeName);
