// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/op/reshape.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::Reshape;
}  // namespace v1
}  // namespace op
}  // namespace ngraph
