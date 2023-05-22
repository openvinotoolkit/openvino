// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include "ngraph/op/util/arithmetic_reductions_keep_dims.hpp"
#include "openvino/op/reduce_l2.hpp"

namespace ngraph {
namespace op {
namespace v4 {
using ov::op::v4::ReduceL2;
}  // namespace v4
}  // namespace op
}  // namespace ngraph
