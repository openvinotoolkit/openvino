// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "openvino/op/softplus.hpp"

namespace ngraph {
namespace op {
namespace v4 {
using ov::op::v4::SoftPlus;
}  // namespace v4
}  // namespace op
}  // namespace ngraph
