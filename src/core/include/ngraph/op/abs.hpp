// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/abs.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::Abs;
}  // namespace v0
using v0::Abs;
}  // namespace op
}  // namespace ngraph
