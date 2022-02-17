// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
