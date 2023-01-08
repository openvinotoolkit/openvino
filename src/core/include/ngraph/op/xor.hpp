// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/util/binary_elementwise_logical.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/xor.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::LogicalXor;
}  // namespace v1
namespace v0 {
using ov::op::v0::Xor;
}  // namespace v0
using v0::Xor;
}  // namespace op
}  // namespace ngraph
