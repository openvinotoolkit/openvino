// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/mod.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::Mod;
}  // namespace v1
}  // namespace op
}  // namespace ngraph
