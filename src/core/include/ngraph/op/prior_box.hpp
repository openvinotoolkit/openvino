// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include "ngraph/op/op.hpp"
#include "openvino/op/prior_box.hpp"

namespace ngraph {
namespace op {
using PriorBoxAttrs = ov::op::v0::PriorBox::Attributes;
namespace v0 {
using ov::op::v0::PriorBox;
}  // namespace v0
namespace v8 {
using ov::op::v8::PriorBox;
}  // namespace v8
using v0::PriorBox;
}  // namespace op
}  // namespace ngraph
