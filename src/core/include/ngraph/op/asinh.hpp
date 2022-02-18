// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/asinh.hpp"

namespace ngraph {
namespace op {
namespace v3 {
using ov::op::v3::Asinh;
}  // namespace v3
using v3::Asinh;
}  // namespace op
}  // namespace ngraph
