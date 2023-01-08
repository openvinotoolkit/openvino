// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/reduction_base.hpp"
#include "openvino/op/util/logical_reduction.hpp"

namespace ngraph {
namespace op {
namespace util {
using ov::op::util::LogicalReduction;
}  // namespace util
}  // namespace op
}  // namespace ngraph
