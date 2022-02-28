// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "ngraph/util.hpp"
#include "openvino/op/sigmoid.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::Sigmoid;
}  // namespace v0
using v0::Sigmoid;
}  // namespace op
}  // namespace ngraph
