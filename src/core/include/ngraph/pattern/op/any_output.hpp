// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

#include "ngraph/node.hpp"
#include "ngraph/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/any_output.hpp"

namespace ngraph {
namespace pattern {
namespace op {
using ov::pass::pattern::op::AnyOutput;
}  // namespace op
}  // namespace pattern
}  // namespace ngraph
