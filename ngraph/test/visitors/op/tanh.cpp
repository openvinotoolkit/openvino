// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "visitors/no_attributes_op.hpp"
#include "ngraph/opsets/opset1.hpp"

using Type = ::testing::Types<opset1::Tanh>;
INSTANTIATE_TYPED_TEST_CASE_P(attributes_tanh, NoAttributesUnaryOp, Type);
