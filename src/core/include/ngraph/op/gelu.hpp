// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/gelu.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::Gelu;
}  // namespace v0
using v0::Gelu;

using ov::op::GeluApproximationMode;

namespace v7 {
using ov::op::v7::Gelu;
}  // namespace v7
}  // namespace op
}  // namespace ngraph
