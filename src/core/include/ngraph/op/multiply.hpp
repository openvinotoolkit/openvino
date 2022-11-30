// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/multiply.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::Multiply;
}  // namespace v1
}  // namespace op
}  // namespace ngraph

#define OPERATION_DEFINED_Multiply 1
#include "ngraph/opsets/opsets_tbl.hpp"
#undef OPERATION_DEFINED_Multiply
