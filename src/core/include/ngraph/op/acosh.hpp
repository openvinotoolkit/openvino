// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

#include <memory>

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/acosh.hpp"

namespace ngraph {
namespace op {
namespace v3 {
using ov::op::v3::Acosh;
}  // namespace v3
using v3::Acosh;
}  // namespace op
}  // namespace ngraph
