// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"
#include "openvino/op/util/op_types.hpp"

namespace ngraph {
namespace op {
using ov::op::util::is_binary_elementwise_arithmetic;
using ov::op::util::is_binary_elementwise_comparison;
using ov::op::util::is_binary_elementwise_logical;
using ov::op::util::is_commutative;
using ov::op::util::is_constant;
using ov::op::util::is_op;
using ov::op::util::is_output;
using ov::op::util::is_parameter;
using ov::op::util::is_sink;
using ov::op::util::is_unary_elementwise_arithmetic;
using ov::op::util::supports_auto_broadcast;
}  // namespace op
}  // namespace ngraph
