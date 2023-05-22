// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "openvino/op/range.hpp"

namespace ngraph {
namespace op {
namespace v4 {
using ov::op::v4::Range;
}  // namespace v4
namespace v0 {
using ov::op::v0::Range;
}  // namespace v0
using v0::Range;
}  // namespace op
}  // namespace ngraph
