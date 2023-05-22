// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include "openvino/op/i420_to_bgr.hpp"

namespace ngraph {
namespace op {
namespace v8 {
using ov::op::v8::I420toBGR;
}  // namespace v8
}  // namespace op
}  // namespace ngraph
