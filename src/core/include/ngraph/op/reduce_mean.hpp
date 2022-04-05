// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_set.hpp"
#include "ngraph/op/util/arithmetic_reductions_keep_dims.hpp"
#include "openvino/op/reduce_mean.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::ReduceMean;
}  // namespace v1
}  // namespace op
}  // namespace ngraph
