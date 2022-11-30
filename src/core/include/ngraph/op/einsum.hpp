// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "openvino/op/einsum.hpp"

namespace ngraph {
namespace op {
namespace v7 {
using ov::op::v7::Einsum;
}  // namespace v7
}  // namespace op
}  // namespace ngraph

#define OPERATION_DEFINED_Einsum 1
#include "ngraph/opsets/opsets_tbl.hpp"
#undef OPERATION_DEFINED_Einsum
