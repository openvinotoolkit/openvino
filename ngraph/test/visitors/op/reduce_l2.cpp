// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_ops.hpp"

using Type = ::testing::Types<ngraph::op::v4::ReduceL2>;
INSTANTIATE_TYPED_TEST_CASE_P(attributes_reduce_l2, ReduceOpsAttrTest, Type);
