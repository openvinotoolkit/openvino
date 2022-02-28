// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/arithmetic_reduction.hpp"
#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"

namespace ngraph {
namespace op {
namespace util {
using ov::op::util::ArithmeticReductionKeepDims;
}  // namespace util
}  // namespace op
}  // namespace ngraph
