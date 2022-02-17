// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/negative.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::Negative;
}  // namespace v0
using v0::Negative;
}  // namespace op
NGRAPH_API
std::shared_ptr<Node> operator-(const Output<Node>& arg0);
}  // namespace ngraph
