// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/logical_reduction_keep_dims.hpp"
#include "openvino/op/reduce_logical_and.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::ReduceLogicalAnd;
}  // namespace v1
}  // namespace op
}  // namespace ngraph
