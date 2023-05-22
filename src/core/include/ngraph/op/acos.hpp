// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

#include <memory>

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/acos.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::Acos;
}  // namespace v0
using v0::Acos;
}  // namespace op
}  // namespace ngraph
