// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_set.hpp"
#include "ngraph/op/op.hpp"
#include "openvino/op/cum_sum.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::CumSum;
}  // namespace v0
using v0::CumSum;
}  // namespace op
}  // namespace ngraph
