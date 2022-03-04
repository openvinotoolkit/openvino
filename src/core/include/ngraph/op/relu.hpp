// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/relu.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::Relu;
}  // namespace v0
using v0::Relu;
}  // namespace op
}  // namespace ngraph
