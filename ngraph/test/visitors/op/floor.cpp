// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unary_ops.hpp"
using Type = ::testing::Types<ngraph::op::v0::Floor>;

INSTANTIATE_TYPED_TEST_CASE_P(visitor_without_atrribute,
                              UnaryOperatorVisitor,
                              Type,
                              UnaryOperatorTypeName);