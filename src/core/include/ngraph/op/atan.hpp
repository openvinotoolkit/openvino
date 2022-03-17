// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/atan.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::Atan;
}  // namespace v0
using v0::Atan;
}  // namespace op
}  // namespace ngraph
